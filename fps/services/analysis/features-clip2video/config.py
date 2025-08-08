from pathlib import Path


class Config:
    def __init__(
        self,
        video_path: str | None = None,
        checkpoint_dir="checkpoint",
        clip_path="checkpoint/ViT-B-32.pt",
        data_path="data/msrvtt_data/MSRVTT_data.json",
    ):
        self.do_eval = True
        self.video_path = Path(__file__).parent / video_path if video_path else None
        self.data_path = Path(__file__).parent / data_path
        self.embeddings_path: str | None = None
        self.num_thread_reader = 0
        self.seed = 42
        self.max_words = 32
        self.max_frames = 36
        self.feature_framerate = 2
        self.output_dir = None
        self.cross_model_name = "cross-base"
        self.do_lower_case = True
        self.num_gpu = 1
        self.cache_dir = ""
        self.fp16 = True
        self.fp16_opt_level = "O1"
        self.cross_num_hidden_layers = 4
        self.sim_type = "seqTransf"
        self.checkpoint_dir = Path(__file__).parent / checkpoint_dir
        self.model_num = 2
        self.local_rank = 0
        self.datatype = "msrvtt"
        self.vocab_size = 49408
        self.temporal_type = "TDB"
        self.temporal_proj = "sigmoid_selfA"
        self.center_type = ""
        self.centerK = 5
        self.center_weight = 0.5
        self.center_proj = "TAB_TDB"
        self.clip_path = Path(__file__).parent / clip_path
        self.gpu = True
