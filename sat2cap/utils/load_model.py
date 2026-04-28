from ..models.geomoco import GeoMoCo
import torch

def load_geomoco(ckpt_path):
    print('Soft Checkpoint Reload')
    checkpoint = torch.load(ckpt_path)
    hparams = checkpoint['hyper_parameters']
    #if this model does not have the geo_encode settings set it to False
    if 'geo_encode' not in hparams:
        hparams['geo_encode'] = False
    hparams['inference'] = True

    geoclip = GeoMoCo(hparams=hparams)
    print('Using Pretrained GeoClip')
    unused_params = geoclip.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f'Unused params {unused_params}')
    return geoclip

def load_clip(ckpt_path):
    print('Soft Checkpoint Reload')
    checkpoint = torch.load(ckpt_path)
    hparams = checkpoint['hyper_parameters']
    hparams['inference'] = True
    clip = GeoMoCo(hparams=hparams)
    print('Using CLIP')
    return clip

def load_sat2cap(repo_id='MVRL/sat2cap', filename='sat2cap.ckpt', ckpt_path=None):
    """Load the Sat2Cap model from HuggingFace Hub or a local checkpoint.

    Args:
        repo_id (str): HuggingFace Hub repository ID. Defaults to 'MVRL/sat2cap'.
        filename (str): Filename of the checkpoint in the repository. Defaults to 'sat2cap.ckpt'.
        ckpt_path (str, optional): Path to a local checkpoint file. If provided,
            the HuggingFace Hub download is skipped.

    Returns:
        GeoMoCo: The loaded Sat2Cap model in evaluation mode.
    """
    if ckpt_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to download the model. "
                "Install it with: pip install huggingface_hub"
            ) from e
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # weights_only=False is required because PyTorch Lightning checkpoints
    # contain Python objects (e.g. hyperparameter dicts) beyond plain tensors.
    # Only load checkpoints from trusted sources such as the official HuggingFace Hub repo.
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = checkpoint['hyper_parameters']
    hparams['inference'] = True
    # spherical_harmonics and dropout_rate must be disabled at inference time:
    # spherical_harmonics changes the geo-encoder architecture and is training-only,
    # while dropout_rate=0 ensures deterministic embeddings without random dropout.
    hparams['spherical_harmonics'] = False
    hparams['dropout_rate'] = 0
    if 'geo_encode' not in hparams:
        hparams['geo_encode'] = False

    model = GeoMoCo(hparams=hparams)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model  