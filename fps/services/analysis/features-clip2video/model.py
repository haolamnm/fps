import torch
import torch.nn.functional as F

from .clip2video.modules.modeling import CLIP2Video as CLIP2VideoBaseModel


class CLIP2VideoModel(CLIP2VideoBaseModel):
    def get_text_features(
        self,
        sequence_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        shaped: bool = False,
    ) -> torch.Tensor:
        if not shaped:
            attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])

        # Handle TAB type, (seq_out, hidden_out)
        if self.center_type == "TAB":
            sequence_output, _ = sequence_output

        if not isinstance(sequence_output, torch.Tensor):
            raise TypeError(f"Expected sequence_output to be a torch.Tensor, but got {type(sequence_output)}")

        sequence_output = sequence_output.squeeze(1).contiguous()
        sequence_output = F.normalize(sequence_output, p=2, dim=-1)
        return sequence_output

    def get_video_features(
        self,
        visual_output: torch.Tensor,
        video_mask: torch.Tensor,
        shaped: bool = False,
    ) -> torch.Tensor:
        if not shaped:
            video_mask = video_mask.reshape(-1, video_mask.shape[-1])

        if not (self.sim_type == "seqTransf" and self.temporal_type == "TDB"):
            raise NotImplementedError("get_video_features is only implemented for seqTransf and TDB temporal type.")

        # Temporal Diffusion Block
        visual_output = visual_output.contiguous()
        visual_output_original = visual_output
        (
            visual_output,
            frame_positional_embeddings,
            type_embedding,
            temporal_video_mask,
        ) = self.temporal_difference_block(visual_output, video_mask)

        # Add frame positional embeddings and type embedding
        visual_output = visual_output + frame_positional_embeddings + type_embedding

        extended_video_mask = (1.0 - temporal_video_mask.unsqueeze(1)) * -1_000_000.0
        extended_video_mask = extended_video_mask.expand(-1, temporal_video_mask.size(1), -1)
        visual_output = self.transformerClip(visual_output.transpose(0, 1), extended_video_mask).transpose(0, 1)

        # Select even frames
        frame_position_id = torch.arange(
            start=0,
            end=visual_output.size()[1],
            step=2,
            dtype=torch.long,
            device=visual_output.device,
        )
        visual_output = visual_output[:, frame_position_id, :] + visual_output_original
        visual_output = F.normalize(visual_output, p=2, dim=-1)

        # Masked mean pooling
        mask_f = video_mask.float().unsqueeze(-1)
        visual_output = (visual_output * mask_f).sum(dim=1)

        denominator = torch.clamp(mask_f.sum(dim=1), min=1.0)
        visual_output = visual_output / denominator

        # Normalize the output
        visual_output = F.normalize(visual_output, p=2, dim=-1)

        return visual_output
