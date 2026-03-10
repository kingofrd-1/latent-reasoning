from contextlib import contextmanager
from types import MethodType
import torch
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors

@contextmanager
def override_forward_with_over_zero(model, factory, insert_embedding, model_name='llama', special_token_id=128025):
    """
    Context manager to temporarily override the model's forward method,
    enabling the insertion of special embeddings during forward pass.

    Args:
        model: The target model object.
        factory: Function that returns a custom forward method.
        insert_embedding: Embeddings to insert for special tokens.
        model_name: Model type ('llama', 'mistral', 'qwen', or 'multi').
        special_token_id: The token ID for the special token.
    """
    # Save the original forward method
    original_forwards = model.llm_engine.model_executor.driver_worker.model_runner.model.model.forward
    # Override forward with the custom function
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.forward = MethodType(
        factory(insert_embedding, model_name, special_token_id),
        model.llm_engine.model_executor.driver_worker.model_runner.model.model
    )
    try:
        yield
    finally:
        # Restore the original forward method after exiting the context
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.forward = original_forwards

def factory(insert_embedding, model_name='llama', special_token_id=128025):
    """
    Factory function to generate different custom forward functions for model types.
    """
    def llama_forward(
        self,
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        intermediate_tensors,
        inputs_embeds,
    ):
        """
        Custom forward for LLaMA/Mistral models to insert special embeddings.
        """
        # Use provided embeddings or get input embeddings from model
        hidden_states = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
        # Find locations of special token
        special_token_index = (input_ids == special_token_id).nonzero(as_tuple=False).squeeze()
        device = hidden_states.device
        embeddings = insert_embedding
        embeddings = [emb[0].to(device) for emb in embeddings]
        # Insert the embeddings at special token positions
        for idx, special_idx in enumerate(special_token_index):
            if idx < len(embeddings):
                hidden_states[special_idx] = embeddings[idx]
        residual = None
        # Run the forward pass through each transformer layer
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i - self.start_layer],
                                            attn_metadata, residual)
        # Final normalization
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def llama_forward_multi(
        self,
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        intermediate_tensors,
        inputs_embeds,
    ):
        """
        Custom forward for multi-special-token insertion in LLaMA/Mistral models.
        """
        hidden_states = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
        special_token_index = (input_ids == special_token_id).nonzero(as_tuple=False).squeeze()
        num_spe_token = int(len(special_token_index) / 5)
        device = hidden_states.device
        embedding_insert = insert_embedding
        embeddings = []
        # Split embeddings into chunks for multiple tokens
        for emb in embedding_insert:
            chunks = torch.chunk(emb[0][0], num_spe_token)
            for chunk in chunks:
                embeddings.append(chunk.to(device))
        # Insert embeddings at the corresponding special token locations
        for idx, special_idx in enumerate(special_token_index):
            if idx < len(embeddings):
                hidden_states[special_idx] = embeddings[idx]
        residual = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i - self.start_layer],
                                            attn_metadata, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def qwen_forward(
        self,
        input_ids,
        positions: torch.Tensor,
        kv_caches,
        attn_metadata,
        intermediate_tensors,
        inputs_embeds=None,
    ):
        """
        Custom forward for Qwen models, handling distributed pipeline parallelism.
        """
        # Only the first pipeline stage handles embedding insertion
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            residual = None
            special_token_index = (input_ids == special_token_id).nonzero(as_tuple=False).squeeze()
            if len(special_token_index.shape) == 0:
                special_token_index = [special_token_index]
            device = hidden_states.device
            embeddings = insert_embedding
            embeddings = [emb[0].to(device) for emb in embeddings]
            for idx, special_idx in enumerate(special_token_index):
                if idx < len(embeddings):
                    hidden_states[special_idx] = embeddings[idx]
        else:
            # Other stages receive hidden_states and residual from previous stage
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        if not get_pp_group().is_last_rank:
            # Pass hidden_states and residual to the next stage
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        # Final normalization at the last stage
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # Choose the correct forward function based on model_name
    if model_name == 'llama' or model_name == 'mistral':
        return llama_forward
    elif model_name == 'qwen':
        return qwen_forward
    elif model_name == "multi":
        return llama_forward_multi
