from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from qr_onboarding.ml_models import QRSyntheticDataset, QRResearchEnhancer, compute_training_loss
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=1); p.add_argument('--samples', type=int, default=64); p.add_argument('--batch-size', type=int, default=8); p.add_argument('--output', default='assets/generated/research_enhancer.pt'); args=p.parse_args()
    ds=QRSyntheticDataset(count=args.samples); dl=DataLoader(ds, batch_size=args.batch_size, shuffle=True); device='cuda' if torch.cuda.is_available() else 'cpu'; model=QRResearchEnhancer().to(device); opt=torch.optim.Adam(model.parameters(), lr=1e-3); model.train()
    for epoch in range(args.epochs):
        total=0.0
        for batch in dl:
            batch={k:v.to(device) for k,v in batch.items()}; out=model(batch['input']); loss=compute_training_loss(out,batch); opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.detach().cpu())
        print({'epoch':epoch+1,'loss':total/max(len(dl),1)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), args.output); print({'saved_to':args.output}); return 0
if __name__=='__main__': raise SystemExit(main())
