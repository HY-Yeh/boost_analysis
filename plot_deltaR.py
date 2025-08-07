import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import vector
import seaborn as sns

def calculate_deltaR(eta1, eta2, phi1, phi2):
    deta = eta1 - eta2
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(deta**2 + dphi**2)

def extract_deltaR(filename):
    events = uproot.open(filename)["Events"]
    pId = events["GenPart_pdgId"].array()
    motherIdx = events["GenPart_genPartIdxMother"].array()
    eta = events["GenPart_eta"].array()
    phi = events["GenPart_phi"].array()

    is_b = np.abs(pId) == 5
    is_W = np.abs(pId) == 24
    is_l = (np.abs(pId) == 11) | (np.abs(pId) == 13)
    is_v = (np.abs(pId) == 12) | (np.abs(pId) == 14)
    valid_mother = (motherIdx >= 0) & (motherIdx < ak.num(pId))
    mother_pId = ak.where(valid_mother, pId[motherIdx], -999)
    
    is_from_top = np.abs(mother_pId) == 6
    is_from_W = np.abs(mother_pId) == 24

    is_b_from_top = is_b & is_from_top
    is_W_from_top = is_W & is_from_top
    is_l_from_W = is_l & is_from_W
    is_v_from_W = is_v & is_from_W

    eta_of_b = eta[is_b_from_top]
    phi_of_b = phi[is_b_from_top]
    eta_of_W = eta[is_W_from_top]
    phi_of_W = phi[is_W_from_top]
    eta_of_l = eta[is_l_from_W]
    phi_of_l = phi[is_l_from_W]
    eta_of_v = eta[is_v_from_W]
    phi_of_v = phi[is_v_from_W]

    eta_of_b = eta_of_b[:, 0]
    phi_of_b = phi_of_b[:, 0]
    eta_of_W = eta_of_W[:, 0]
    phi_of_W = phi_of_W[:, 0]
    eta_of_l = eta_of_l[:, 0]
    phi_of_l = phi_of_l[:, 0]
    eta_of_v = eta_of_v[:, 0]
    phi_of_v = phi_of_v[:, 0]
    

    deltaR = calculate_deltaR(eta_of_b, eta_of_v, phi_of_b, phi_of_v)

    return deltaR

mass_points = [700, 800, 900, 1000, 1500, 2000, 2500, 3000]
filepaths = [
    "/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_700_rtt06_rtc04.root",
    "/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_800_rtt06_rtc04.root",
    "/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_900_rtt06_rtc04.root",
    "/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_1000_rtt06_rtc04.root",
    "/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_1500_rtt06_rtc04.root",
    "/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_2000_rtt06_rtc04.root",
    "/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_2500_rtt06_rtc04.root",
    "/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_3000_rtt06_rtc04.root",
]

all_masses = []
all_dRs = []

for mass, file in zip(mass_points, filepaths):
    dRs = extract_deltaR(file)
    all_masses.extend([mass] * len(dRs))
    all_dRs.extend(dRs)


sns.set(style="whitegrid")

plt.figure(figsize=(8,6))
sns.violinplot(x=all_masses, y=all_dRs, palette="muted", inner="quartile")
plt.xlabel("mH⁺ [GeV]")
plt.ylabel("ΔR(b, v)")
plt.title("ΔR(b, v) vs H⁺ highmass")
plt.axhline(0.4, color='red', linestyle='--', linewidth=0.5)
plt.axhline(0.8, color='red', linestyle='--', linewidth=0.5)
plt.yticks(np.arange(0, 3.6, 0.4))
plt.ylim(0, 3.2)
plt.tight_layout()
plt.savefig('plot/bv_deltaR_highmass.png', dpi=300)
print("density plot finished!")

'''
all_masses = np.array(all_masses)
all_dRs = np.array(all_dRs)

bins = np.linspace(0, 5, 20)
color_list = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'brown', 'gray']
colors = {mass: color_list[i % len(color_list)] for i, mass in enumerate(mass_points)}

plt.figure(figsize=(8,6))
    
for mass in mass_points:
    mask = all_masses == mass
    dRs_mass = all_dRs[mask]

    counts, edges = np.histogram(dRs_mass, bins=bins)
    probabilities = counts / counts.sum()

    plt.bar(edges[:-1], probabilities, width=np.diff(edges), 
            alpha=0.5, label=f"{mass} GeV", color=colors[mass], align='edge')


plt.xlabel("ΔR(l, v)")
plt.ylabel("Normalized Events (1/N)")
plt.title("ΔR(l, v) Distribution for Different H⁺ Masses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot/lv_deltaR_bin.png', dpi=300)
'''