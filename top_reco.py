import uproot
import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector
import matplotlib.pyplot as plt

events_500 = uproot.open("/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_500_rtt06_rtc04.root")["Events"]
events_1000 = uproot.open("/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_1000_rtt06_rtc04.root")["Events"]
#events_1500 = uproot.open("/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_1500_rtt06_rtc04.root")["Events"]
#events_2000 = uproot.open("/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_2000_rtt06_rtc04.root")["Events"]
#events_3000 = uproot.open("/eos/cms/store/group/phys_b2g/ExYukawa/bHplus/2017/v7/CGToBHpm_a_3000_rtt06_rtc04.root")["Events"]


def mass_reconstruction(events):

    print(events)  

    #################
    ### Selection ###
    #################

    tightElectrons_id = events["tightElectrons_noIso_id"].array()
    valid_ele_idx = tightElectrons_id[tightElectrons_id != -1]

    tightMuons_id = events["tightMuons_noIso_id"].array()
    valid_muon_idx = tightMuons_id[tightMuons_id != -1]    


    valid_lepton_idx = ak.concatenate([valid_ele_idx, valid_muon_idx], axis=1)
    lepton_mask = ak.sum(valid_lepton_idx >= 0, axis=1) == 1

    '''
    btag = events["Jet_btagDeepB"].array()
    btag = btag[lepton_mask]
    sorted_btag = ak.argsort(btag, axis=1, ascending=False)
    second_b = btag[sorted_btag]
    mask_2b = second_b[:, 1] > 0.6
    '''

    tightbJets_id = events["tightJets_b_DeepJetmedium_id"].array()
    tightbJets_id = tightbJets_id[lepton_mask]
    valid_bjet_idx = tightbJets_id[tightbJets_id != -1]
    mask_2b = ak.sum(valid_bjet_idx >= 0, axis=1) == 2

    Electron_pt = events["Electron_pt"].array()
    Electron_pt = Electron_pt[valid_ele_idx]
    Electron_pt = Electron_pt[lepton_mask]

    Muon_pt = events["Muon_pt"].array()
    Muon_pt = Muon_pt[valid_muon_idx]
    Muon_pt = Muon_pt[lepton_mask]
    
    
    l_pt = ak.concatenate([Electron_pt, Muon_pt], axis=1)
    sorted_lpt = ak.argsort(l_pt, axis=1, ascending=False)
    l_pt = l_pt[sorted_lpt]
    l_pt = l_pt[mask_2b]
    #print(l_pt)
    mask_lpt = l_pt[:, 0] > 25


    Electron_eta = events["Electron_eta"].array()
    Electron_eta = Electron_eta[valid_ele_idx]
    Electron_eta = Electron_eta[lepton_mask]
    
    Muon_eta = events["Muon_eta"].array()
    Muon_eta = Muon_eta[valid_muon_idx]
    Muon_eta = Muon_eta[lepton_mask]
    
    l_eta = ak.concatenate([Electron_eta, Muon_eta], axis=1)
    l_eta = l_eta[sorted_lpt]
    l_eta = l_eta[mask_2b]
    l_eta = l_eta[mask_lpt]
    mask_leta = np.abs(l_eta[:, 0]) < 2.4


    Jet_pt = events["Jet_pt"].array()
    Jet_pt = Jet_pt[lepton_mask]
    sorted_pt = ak.argsort(Jet_pt, axis=1, ascending=False)
    Jet_pt = Jet_pt[sorted_pt]
    Jet_pt = Jet_pt[mask_2b]
    Jet_pt = Jet_pt[mask_lpt]
    Jet_pt = Jet_pt[mask_leta]
    mask_jetpt = Jet_pt[:, 0] > 30

    #########################
    ### Pz reconstruction ###
    #########################

    Electron_pt = events["Electron_pt"].array()
    Electron_pt = Electron_pt[valid_ele_idx]
    Electron_pt = Electron_pt[lepton_mask]
    
    Muon_pt = events["Muon_pt"].array()
    Muon_pt = Muon_pt[valid_muon_idx]
    Muon_pt = Muon_pt[lepton_mask]
    
    l_pt = ak.concatenate([Electron_pt, Muon_pt], axis=1)
    l_pt = l_pt[mask_2b]
    l_pt = l_pt[mask_lpt]
    l_pt = l_pt[mask_leta]
    l_pt = l_pt[mask_jetpt]


    Electron_eta = events["Electron_eta"].array()
    Electron_eta = Electron_eta[valid_ele_idx]
    Electron_eta = Electron_eta[lepton_mask]
    
    Muon_eta = events["Muon_eta"].array()
    Muon_eta = Muon_eta[valid_muon_idx]
    Muon_eta = Muon_eta[lepton_mask]
    
    l_eta = ak.concatenate([Electron_eta, Muon_eta], axis=1)
    l_eta = l_eta[mask_2b]
    l_eta = l_eta[mask_lpt]
    l_eta = l_eta[mask_leta]
    l_eta = l_eta[mask_jetpt]

    Electron_phi = events["Electron_phi"].array()
    Electron_phi = Electron_phi[valid_ele_idx]
    Electron_phi = Electron_phi[lepton_mask]
    
    Muon_phi = events["Muon_phi"].array()
    Muon_phi = Muon_phi[valid_muon_idx]
    Muon_phi = Muon_phi[lepton_mask]
    
    l_phi = ak.concatenate([Electron_phi, Muon_phi], axis=1)
    l_phi = l_phi[mask_2b]
    l_phi = l_phi[mask_lpt]
    l_phi = l_phi[mask_leta]
    l_phi = l_phi[mask_jetpt]

    MET = events["MET_pt"].array()
    MET = MET[lepton_mask]
    MET = MET[mask_2b]
    MET = MET[mask_lpt]
    MET = MET[mask_leta]
    MET = MET[mask_jetpt]

    MET_phi = events["MET_phi"].array()
    MET_phi = MET_phi[lepton_mask]
    MET_phi = MET_phi[mask_2b]
    MET_phi = MET_phi[mask_lpt]
    MET_phi = MET_phi[mask_leta]
    MET_phi = MET_phi[mask_jetpt]


    mW = 80.4
    Lambda = mW*mW/2. + (l_pt * MET * np.cos(l_phi - MET_phi))
    l_pz = l_pt * np.sinh(l_eta)
    l_E  = np.sqrt(l_pz*l_pz + l_pt*l_pt)
    D = Lambda*Lambda*l_pz*l_pz + l_pt*l_pt*(Lambda*Lambda - l_E * l_E * MET * MET)
    A = Lambda*l_pz/(l_pt*l_pt)

    if ak.any(D < 0):
      MET_pz = A
    else:
      if np.abs(A + np.sqrt(D)/(l_pt*l_pt)) > np.abs(A - np.sqrt(D)/(l_pt*l_pt)):
        MET_pz = A - np.sqrt(D)/(l_pt*l_pt)
      else:
        MET_pz = A + np.sqrt(D)/(l_pt*l_pt)

    Neutrino = ak.zip(
      {
        "pt":  MET,
        "eta": np.arcsinh(MET_pz / MET),
        "phi": MET_phi,
        "mass": 0,
      },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
    )

    #######################
    ### reco level info ###
    #######################

    Jet_pt = events["Jet_pt"].array()
    Jet_pt = Jet_pt[lepton_mask]
    Jet_pt = Jet_pt[mask_2b]
    Jet_pt = Jet_pt[mask_lpt]
    Jet_pt = Jet_pt[mask_leta]
    Jet_pt = Jet_pt[mask_jetpt]
    
    Jet_eta = events["Jet_eta"].array()
    Jet_eta = Jet_eta[lepton_mask]
    Jet_eta = Jet_eta[mask_2b]
    Jet_eta = Jet_eta[mask_lpt]
    Jet_eta = Jet_eta[mask_leta]
    Jet_eta = Jet_eta[mask_jetpt]

    Jet_phi = events["Jet_phi"].array()
    Jet_phi = Jet_phi[lepton_mask]
    Jet_phi = Jet_phi[mask_2b]
    Jet_phi = Jet_phi[mask_lpt]
    Jet_phi = Jet_phi[mask_leta]
    Jet_phi = Jet_phi[mask_jetpt]

    Jet_mass = events["Jet_mass"].array()
    Jet_mass = Jet_mass[lepton_mask]
    Jet_mass = Jet_mass[mask_2b]
    Jet_mass = Jet_mass[mask_lpt]
    Jet_mass = Jet_mass[mask_leta]
    Jet_mass = Jet_mass[mask_jetpt]

    
    sorted_pt = ak.argsort(Jet_pt, axis=1, ascending=False)
    Jet_pt = Jet_pt[sorted_pt]
    Jet_eta = Jet_eta[sorted_pt]
    Jet_phi = Jet_phi[sorted_pt]
    Jet_mass = Jet_mass[sorted_pt]
    
    
    hardJet_0 = ak.zip(
      {
        "pt":   Jet_pt[:, 0],
        "eta":  Jet_eta[:, 0],
        "phi":  Jet_phi[:, 0],
        "mass": Jet_mass[:, 0],
      },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
    )
  

    hardJet_1 = ak.zip(
      {
        "pt":   Jet_pt[:, 1],
        "eta":  Jet_eta[:, 1],
        "phi":  Jet_phi[:, 1],
        "mass": Jet_mass[:, 1],
      },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
    )

    
    lepton = ak.zip(
      {
        "pt":   l_pt,
        "eta":  l_eta,
        "phi":  l_phi,
        "mass": 0,
      },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
    )

    ########################
    ### calculate deltaR ###
    ########################

    def calculate_deltaR(eta1, eta2, phi1, phi2):
      deta = eta1 - eta2
      dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
      return np.sqrt(deta**2 + dphi**2)

    deltaR_0 = calculate_deltaR(Jet_eta[:, 0], l_eta[:, 0], Jet_phi[:, 0], l_phi[:, 0])
    print("deltaR_0 :" ,deltaR_0)
    deltaR_1 = calculate_deltaR(Jet_eta[:, 1], l_eta[:, 0], Jet_phi[:, 1], l_phi[:, 0])
    print("deltaR_1 :" ,deltaR_1)
    
    dr_0 = lepton.delta_r(hardJet_0)
    print("dr_0 :",dr_0)
    dr_1 = lepton.delta_r(hardJet_1)
    print("dr_1 :",dr_1)


    ##########################
    ### rop reconstruction ###
    ##########################

    top_mass_0 = ak.where(deltaR_0 > 0.4, (hardJet_0 + lepton[:, 0] + Neutrino[:, 0]).mass, (hardJet_0 + Neutrino[:, 0]).mass)
    #print("top_mass_0 :" ,top_mass_0)
    top_mass_1 = ak.where(deltaR_1 > 0.4, (hardJet_1 + lepton[:, 0] + Neutrino[:, 0]).mass, (hardJet_1 + Neutrino[:, 0]).mass)
    #print("top_mass_1 :" ,top_mass_1)

    top_mass = ak.where(np.abs(172.76 - top_mass_0) < np.abs(172.76 - top_mass_1),
                                top_mass_0, top_mass_1)
    mask_above4 = ((np.abs(172.76 - top_mass_0) < np.abs(172.76 - top_mass_1)) & (deltaR_0 > 0.4)) | \
               ((np.abs(172.76 - top_mass_0) >= np.abs(172.76 - top_mass_1)) & (deltaR_1 > 0.4))

    mask_under4 = ~mask_above4
    
    #top_mass = top_mass[mask_under4]
                               
    return top_mass, deltaR_0, deltaR_1


events_dict = {
    500: events_500,
    1000: events_1000,
    #1500: events_1500,
    #2000: events_2000,
    #3000: events_3000
}

for mass, events in events_dict.items():

  top_mass, deltaR_0, deltaR_1 = mass_reconstruction(events)
  print("top_mass :", top_mass)

  plt.figure(figsize=(8, 6))
  plt.hist(top_mass, bins=40, range=(0, 1000), histtype='step', color='blue', linewidth=1.5)
  plt.xlabel("Reconstructed top mass [GeV]")
  plt.ylabel("Events")
  plt.title(f"Top Mass Reconstruction for H+ {mass}GeV when ΔR < 0.4")
  plt.xticks(np.arange(0, 1025, 50))
  plt.axvline(x=173, color='red', linestyle='--', linewidth=1.2)
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(f"plot/top_mass {mass}GeV under 0.4.png", dpi=300)
  plt.close()

  '''
  plt.figure(figsize=(8, 6))
  plt.hist(deltaR_0, bins=20, range=(0, 5), histtype='step', color='blue', linewidth=1.5)
  plt.xlabel("DeltaR [GeV]")
  plt.ylabel("ΔR(b1, l)")
  plt.title(f"ΔR(b1, l) distribution for H+ {mass}GeV")
  plt.xticks(np.arange(0, 6, 1))
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(f"plot/bl1_deltaR reco level at {mass}GeV.png", dpi=300)
  plt.close()

  plt.figure(figsize=(8, 6))
  plt.hist(deltaR_1, bins=20, range=(0, 5), histtype='step', color='blue', linewidth=1.5)
  plt.xlabel("DeltaR [GeV]")
  plt.ylabel("ΔR(b2, l)")
  plt.title(f"ΔR(b2, l) distribution for H+ {mass}GeV")
  plt.xticks(np.arange(0, 6, 1))
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(f"plot/bl2_deltaR reco level at {mass}GeV.png", dpi=300)
  plt.close()
  '''

  