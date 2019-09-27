import torch
from torch.autograd import Variable

def conjugate_gradients(Avp, a, b, model_inner, optimizer_G_inner, nsteps, meta_lambda, residual_tol=1e-10, cuda = False):
    x = torch.zeros(b.size())
    if cuda:
        x = x.cuda()
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(a, p, model_inner, optimizer_G_inner, meta_lambda)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def Fvp(u, s, model_inner, optimizer_G_inner, meta_lambda):
    u.requires_grad_(True)
    kl_v = (u * Variable(s)).sum()
    kl_v.backward()
    flat_grad_grad_kl = []
    for model_param in model_inner.parameters():
        flat_grad_grad_kl.append(model_param.grad.view(-1))
    flat_grad_grad_kl = torch.cat(flat_grad_grad_kl)
    optimizer_G_inner.zero_grad()

    return flat_grad_grad_kl/meta_lambda + Variable(s)