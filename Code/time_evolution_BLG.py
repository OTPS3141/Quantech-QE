import numpy as np
import matplotlib . pyplot as plt
import matplotlib
import scipy as sci
# matplotlib . rc ( ’ xtick ’ , labelsize =20)
# matplotlib . rc ( ’ ytick ’ , labelsize =20)
# ! pip install qutip
import numpy as np
from qutip import *
from tqdm import tqdm
import matplotlib . colors as mcolors
bar = ui.progressbar.EnhancedTextProgressBar(chunk_size=0.1)

M = 2
N = 2*M

zero = tensor([Qobj(np.zeros((2,2))) for i in range(N)])

fs = [tensor([create(2) if i == j else Qobj(np.zeros((2,2))) for i in range(N)]) for j in range(N)]

def sumj(j):

    s = tensor([Qobj(np.zeros((2,2))) for i in range(N)])
    i = 0
    while i < j:
        s = s + fs[i].dag()*fs[i]
        i += 1
    
    return s

cs = [(1j*np.pi*sumj(j)).expm()*fs[j] for j in range(N)]
ns = [c*c.dag() for c in cs]

longc=cs.copy()
longn=ns.copy() 

cs = [[cs[2*i], cs[2*i+1]] for i in range(M)]
ns = [[ns[2*i], ns[2*i+1]] for i in range(M)]

Num = 0 
for n in longn:
    Num += n

# Vacuum state
A = np.array([c.dag() for c in longc]).reshape(N*2**N, 2**N)

vac = Qobj((sci.linalg.null_space(A)).reshape((2**N, )), shape(2**N,1), dims=[[2]*N, [1]*N], type='ket')
# Spin operators
sx = [cs[i][0]*cs[i][1].dag() + cs[i][1]*cs[i][0].dag() for i in range(M)]
sy = [1j*(-cs[i][0]*cs[i][1].dag() + cs[i][1]*cs[i][0].dag()) for i in range(M)]
sz = [cs[i][0]*cs[i][0].dag() - cs[i][1]*cs[i][1].dag() for i in range(M)]

# Location operators
lx = [cs[0][i].cs[1][i].dag() + cs[1][i].cs[0][i].dag() for i in range(2)]
ly = [1j*(-cs[0][i].cs[1][i].dag() + cs[1][i].cs[0][i].dag()) for i in range(2)]
lz = [cs[0][i].cs[0][i].dag() - cs[1][i].cs[1][i].dag() for i in range(2)]

# Computational basis
c = {'Tu': cs[0][0], 'Td': cs[0][1], 'Bu': cs[1][0], 'Bd': cs[1][1]}

# Total basis states
Basis = [vac, c['Tu']*vac, c['Td']*vac, c['Bu']*vac, c['Bd']*vac, c['Tu']*c['Td']*vac, c['Bu']*c['Bd']*vac, c['Tu']*c['Bu']*vac, 
         c['Tu']*c['Bd']*vac, c['Td']*c['Bu']*vac, c['Td']*c['Bd']*vac, c['Tu']*c['Td']*vac/2 - c['Td']*c['Tu']*vac/2, c['Bu']*c['Bd']*vac/2 - c['Bd']*c['Bu']*vac/2,
         c['Tu']*c['Td']*c['Bu']*vac/2 - c['Td']*c['Tu']*c['Bu']*vac/2, c['Tu']*c['Td']*c['Bd']*vac/2 - c['Td']*c['Tu']*c['Bd']*vac/2, c['Tu']*c['Bu']*c['Bd']*vac/2 - c['Tu']*c['Bd']*c['Bu']*vac/2,
         c['Td']*c['Bu']*c['Bd']*vac/2 - c['Td']*c['Bd']*c['Bu']*vac/2, c['Tu']*c['Td']*c['Bu']*c['Bd']*vac/2 - c['Td']*c['Tu']*c['Bu']*c['Bd']*vac/2, 
         c['Tu']*c['Td']*(c['Bu']*c['Bd']*vac/2 - c['Bd']*c['Bu']*vac/2)/2 - c['Td']*c['Tu']*(c['Bu']*c['Bd']*vac/2 - c['Bd']*c['Bu']*vac/2)/2]

ket = { '00':Basis[0],
        'u0':Basis[1],
        'd0':Basis[2],
        '0u':Basis[3],
        '0d':Basis[4],
        'uu':Basis[5],
        'ud':Basis[6],
        'du':Basis[7],
        'dd':Basis[8],
        'S0':Basis[9],
        '0S':Basis[10],
        'Su':Basis[11],
        'Sd':Basis[12],
        'uS':Basis[13],
        'dS':Basis[14],
        'Suu':Basis[15],
}

bra = {s:ket[s].dag() for s in ket}

# Hamiltonian

func = lambda x:x
foo = type(func)

def Hsum(Hs, t_dep=False):
    
    Hfuncs = [h for h in Hs if type(h) == foo]
    Hconst = [h for h in Hs if type(h) != foo]
    H = zero
    for Hi in Hconst:
        H += Hi
    if Hfuncs == [] and not t_dep:
        return H
    else:
        def out(t, args):
            Ht = zero
            for h in [h(t, args) for h in Hfuncs]:
                Ht += h
            return Ht + H
        return out
    
# Rotating frame
def evolve(s, t, op, hbar=1):
    assert type(op) == type(zero)
    assert type(s) == type(zero)
    U=(1j*op*t/hbar).expm()
    if s.type == 'oper':
        return U*s
    if s.type == 'ket':
        return U*s
    if s.type == 'bra':
        return s*U.dag()
    
ketbra = lambda s:ket[s]*bra[s]

# Total spin operators
Sx, Sy, Sz = zero, zero, zero
for i in range(M):
    Sx += sx[i]
    Sy += sy[i]
    Sz += sz[i]

# Total location operators
Lx, Ly, Lz = zero, zero, zero
for i in range(2):
    Lx += lx[i]
    Ly += ly[i]
    Lz += lz[i]

Loc = zero
for i in range(M):
    for j in range(2):
        Loc += (-1)**i*cs[i][j]*cs[i][j].dag()

# Energies of basis states
E1 = 1
E2 = E1 + 0.03
E3 = -E1
E4 = -E1 + 0.03

# Tunneling rate
t = 2

# SOC
SOT = 3
SOB = 0

# g-factor
landau = 2

e_ov_pl = 1519267.4605 # C /( J * ns )

def H_chargesplit(E):
    if type(E) == foo:
        return E*Lz
    else:
        return lambda t, args:E(t)*Lz
    
def H_spinsplit(E):
    return E*Sz

def H_EDSR(alpha, w, w0, Ampl, phi=0):
    if type(alpha) != type((0,0)):
        return lambda t, args:2*alpha*Ampl*e_ov_pl*w/w0**2*Sx*np.cos(w*t + phi)
    else:
        return lambda t, args:2*(alpha[0]*sx[0] + alpha[1]*sx[1])*Ampl*e_ov_pl*w/w0**2*np.cos(w*t + phi)
    
def H_LZSI(eps, w, phi=0):
    if type(eps) != foo:
        return lambda t, args: eps*Lx
    else:
        return lambda t, args: eps(t)*Lx
    
def subspace(H, basis=None):
    
    return np.array([[bi.dag()*H*bj for bi in basis] for bj in basis]).reshape(len(basis), len(basis))

  
# Parameters

E_charge = -800 #ueV
eps = 10.7 #ueV

alpha = (0, 2000) #ueV*A
Efield = 3e-7 #V/m

mu_bohr = 5.7883818012e-5 #ueV/T
planck = 4.135667696e-15 #eV*s
m_e_c2 = 510998.9461 #ueV
c2 = (299792458)**2 #m^2/s^2
r0 = 20 #nm
B = 0.2 #T
w0 = planck*c2/(0.05*m_e_c2*r0**2) #ueV
w_spin = w0*0.0005 #GHz

E_spin = (landau*mu_bohr*B)/2 * 1e6 #ueV

hbar = 6.58211e2 #ueV*ns

# Driving pulse time
on1 = 0/hbar
T1 = 0
on2 = 0/hbar
T2 = 100/hbar
chill = 20/hbar

T = on1 + T1 + on2 + T2 + chill

w_charge = E_charge*2

omega_R = 2*2*alpha[0]*Efield*e_ov_pl*w_spin/w0**2/hbar*1e-3 #GHz

# Time evolution

def H(t, arg):

    if t<0:
        return zero
    if t<on1+T1:
        return Hsum((H_spinsplit(E_spin), H_chargesplit(E_charge), H_LZSI(eps, w_charge)), t_dep=True)(t, arg)
    if t<on1+T1+on2+T2:
        return Hsum((H_spinsplit(E_spin), H_chargesplit(E_charge), H_LZSI(eps, w_charge), H_EDSR(alpha, w_spin, w0, Efield)), t_dep=True)(t, arg)
    if t<=T:
        return Hsum((H_spinsplit(E_spin), H_chargesplit(E_charge), H_LZSI(eps, w_charge)), t_dep=True)(t, arg)
    if t>T:
        return zero
    
# Ground state
psi0=(ket['0u'])
N_steps=300
tlist = np.linspace(0, T, N_steps)

result = mesolve(H, psi0, tlist, progress_bar=bar, args=(), options=Options(store_states=True))

print(result.expect[0][-1])