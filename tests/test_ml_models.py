from qr_onboarding.ml_models import QRSyntheticDataset, QRResearchEnhancer, compute_training_loss
def test_ml_model_forward_shapes_and_trainability():
    ds=QRSyntheticDataset(count=2,image_size=64); batch=ds[0]; model=QRResearchEnhancer(base=8); out=model(batch['input'].unsqueeze(0)); assert out['segmentation'].shape[-2:]==(64,64); assert out['deblurred'].shape[-2:]==(64,64); assert out['super_res'].shape[-2:]==(128,128); loss=compute_training_loss(out,{k:v.unsqueeze(0) for k,v in batch.items() if k!='input'}); loss.backward(); assert any(p.grad is not None for p in model.parameters())
