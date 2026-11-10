import torch
import torch.nn.functional as F


def info_nce(z1, z2, temperature=0.2):
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("z1 and z2 must have the same 2D shape")

    if z1.size(0) < 2 or temperature <= 0:
        raise ValueError("Invalid batch size or temperature")

    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    similarity = z @ z.T
    logits = similarity / temperature

    mask = torch.eye(
        2 * batch_size,
        device=z.device,
        dtype=torch.bool
    )

    logits = logits.masked_fill(mask, float("-inf"))

    labels = torch.arange(
        2 * batch_size, device=z.device
    )

    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(logits, labels)

    return loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)

    z1 = torch.randn(8, 128, device=device, requires_grad=True)
    z2 = torch.randn(8, 128, device=device, requires_grad=True)

    loss = info_nce(z1, z2, temperature=0.2)

    print("Device:", device)
    print("InfoNCE loss:", loss.item())

    loss.backward()

    print("Gradient z1:", z1.grad.norm().item())
    print("Gradient z2:", z2.grad.norm().item())

    assert torch.isfinite(loss)
    assert torch.isfinite(z1.grad).all()
    assert torch.isfinite(z2.grad).all()

    a = torch.eye(8, device=device)
    b = torch.eye(8, device=device)

    good_loss = info_nce(a, b)
    bad_loss = info_nce(a, b.roll(1, dims=0))

    print("Aligned pairs:", good_loss.item())
    print("Incorrect pairs:", bad_loss.item())

    assert good_loss < bad_loss

    print("InfoNCE: OK")
