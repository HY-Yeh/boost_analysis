import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import run_uproot_job, futures_executor
from coffea.nanoevents.methods import vector
import optparse, argparse
import glob
import matplotlib.pyplot as plt


class MySelectionProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):

        ####################
        ### Gen matching ###
        ####################

        Electron = events.Electron
        ele_pt = Electron.pt
        Electron["idx"] = ak.argsort(ak.argsort(Electron.pt), ascending=False)
        Electron["v4"] = ak.zip(
            {
                "pt":   Electron.pt,
                "eta":  Electron.eta,
                "phi":  Electron.phi,
                "mass": Electron.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        Muon = events.Muon
        mu_pt = Muon.pt
        Muon["idx"] = ak.argsort(ak.argsort(Muon.pt), ascending=False)
        Muon["v4"] = ak.zip(
            {
                "pt":   Muon.pt,
                "eta":  Muon.eta,
                "phi":  Muon.phi,
                "mass": Muon.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        GenPart = events.GenPart
        pId = GenPart.pdgId
        is_ele = abs(pId) == 11
        is_mu = abs(pId) == 13

        GenElectron = GenPart[is_ele]
        GenElectron = GenElectron[GenElectron.status == 1]
        GenElectron["idx"] = ak.argsort(ak.argsort(GenElectron.pt), ascending=False)
        GenElectron["v4"] = ak.zip(
            {
                "pt":   GenElectron.pt,
                "eta":  GenElectron.eta,
                "phi":  GenElectron.phi,
                "mass": GenElectron.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        GenMuon = GenPart[is_mu]
        GenMuon = GenMuon[GenMuon.status == 1]
        GenMuon["idx"] = ak.argsort(ak.argsort(GenMuon.pt), ascending=False)
        GenMuon["v4"] = ak.zip(
            {
                "pt":   GenMuon.pt,
                "eta":  GenMuon.eta,
                "phi":  GenMuon.phi,
                "mass": GenMuon.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )


        def gen_matching(candidate, target, dr_cut):
            best_candidate = target.nearest(candidate)
            #print("best_candidate idx:",best_candidate.idx)
            candidate_index = ak.where(target.delta_r(best_candidate) < dr_cut, best_candidate.idx, -1)
            return candidate_index
        

        Electron_idx = gen_matching(Electron, GenElectron.v4, 0.4)
        Muon_idx = gen_matching(Muon, GenMuon.v4, 0.4)

        candidate_reco_electron = Electron[Electron_idx]
        candidate_reco_muon = Muon[Muon_idx]
        
        lepton = ak.concatenate([candidate_reco_electron, candidate_reco_muon], axis=1)
        mask_1l = (ak.num(lepton) == 1)
        
        events_1l = events[mask_1l]
        reco_electron = candidate_reco_electron[mask_1l]
        reco_muon = candidate_reco_muon[mask_1l]

        electron_pt = reco_electron.pt
        muon_pt = reco_muon.pt
        abs_electron_eta = abs(reco_electron.eta)
        abs_muon_eta = abs(reco_muon.eta)
        electron_iso = reco_electron.miniPFRelIso_all
        muon_iso = reco_muon.miniPFRelIso_all

        #######################
        ### LHE information ###
        #######################

        pdgId = events_1l.LHEPart.pdgId

        is_e = abs(pdgId) == 11
        is_mu = abs(pdgId) == 13

        has_e = ak.any(is_e, axis=1)
        has_mu = ak.any(is_mu, axis=1)

        mask_e_only = has_e & ~has_mu
        mask_mu_only = has_mu & ~has_e
        mask_lhe_1l = mask_e_only | mask_mu_only

        lhe_lepton = events_1l[mask_lhe_1l]
        lhe_electron = events_1l[mask_e_only]
        lhe_muon = events_1l[mask_mu_only]

        lhe_ele_pt = ak.flatten(lhe_electron.LHEPart[abs(lhe_electron.LHEPart.pdgId) == 11].pt, axis=None)
        lhe_mu_pt  = ak.flatten(lhe_muon.LHEPart[abs(lhe_muon.LHEPart.pdgId) == 13].pt, axis=None)
        
        lhe_ele_pt = ak.to_list(lhe_ele_pt)
        lhe_mu_pt  = ak.to_list(lhe_mu_pt)

        #############################
        ### Electron WP selection ###
        #############################
        
        pass_WPL = reco_electron.mvaFall17V2noIso_WPL
        pass_WP90 = reco_electron.mvaFall17V2noIso_WP90
        pass_WP80 = reco_electron.mvaFall17V2noIso_WP80

        fail_WP90 = ~pass_WP90
        fail_WP80 = ~pass_WP80

        ########################
        ### Lepton selection ###
        ########################
        

        tight_electron_mask = (
            ((abs_electron_eta < 1.4442) | ((1.566 < abs_electron_eta) & (abs_electron_eta < 2.5)))
            & pass_WP80
            #& (reco_electron.miniPFRelIso_all < 0.1)
            & (has_e == True)
        )
        tight_electron_mask = ak.fill_none(tight_electron_mask, False)
        tight_electron = reco_electron[tight_electron_mask]
        
        medium_electron_mask = (
            ((abs_electron_eta < 1.4442) | ((1.566 < abs_electron_eta) & (abs_electron_eta < 2.5)))
            & pass_WP90
            #& (reco_electron.miniPFRelIso_all < 0.1)
            & (has_e == True)
        )
        medium_electron_mask = ak.fill_none(medium_electron_mask, False)
        medium_electron = reco_electron[medium_electron_mask]

        loose_electron_not_tight_mask = (           
            ((abs_electron_eta < 1.4442) | ((1.566 < abs_electron_eta) & (abs_electron_eta < 2.5)))
            #& (reco_electron.pt > 35)
            & pass_WPL
            & fail_WP80
            #& (reco_electron.miniPFRelIso_all < 0.4)
            & (has_e == True)
        )
        loose_electron_not_tight_mask = ak.fill_none(loose_electron_not_tight_mask, False)
        loose_electron_not_tight = reco_electron[loose_electron_not_tight_mask]
        
        loose_electron_not_medium_mask = (          
            ((abs_electron_eta < 1.4442) | ((1.566 < abs_electron_eta) & (abs_electron_eta < 2.5)))
            #& (reco_electron.pt > 35)
            & pass_WPL
            & fail_WP90
            #& (reco_electron.miniPFRelIso_all < 0.4)
            & (has_e == True)
        )
        loose_electron_not_medium_mask = ak.fill_none(loose_electron_not_medium_mask, False)
        loose_electron_not_medium = reco_electron[loose_electron_not_medium_mask]

        tight_muon_mask = (       
            (abs_muon_eta < 2.4)
            & (reco_muon.tightId)
            #& (reco_muon.miniPFRelIso_all < 0.1)
            & (has_mu == True)
        )
        tight_muon_mask = ak.fill_none(tight_muon_mask, False)
        tight_muon = reco_muon[tight_muon_mask]
        
        medium_muon_mask = (
            (abs_muon_eta < 2.4)
            & (reco_muon.mediumId)
            #& (reco_muon.miniPFRelIso_all < 0.1)
            & (has_mu == True)
        )
        medium_muon_mask = ak.fill_none(medium_muon_mask, False)
        medium_muon = reco_muon[medium_muon_mask]
        
        loose_muon_not_tight_mask = (        
            (abs_muon_eta < 2.4)
            #& (reco_muon.pt > 35)
            & (reco_muon.looseId)
            & (~reco_muon.tightId)
            #& (reco_muon.miniPFRelIso_all < 0.4)
            & (has_mu == True)
        )
        loose_muon_not_tight_mask = ak.fill_none(loose_muon_not_tight_mask, False)
        loose_muon_not_tight = reco_muon[loose_muon_not_tight_mask]
        
        loose_muon_not_medium_mask = (       
            (abs_muon_eta < 2.4)
            #& (reco_muon.pt > 35)
            & (reco_muon.looseId)
            & (~reco_muon.mediumId)
            #& (reco_muon.miniPFRelIso_all < 0.4)
            & (has_mu == True)
        )
        loose_muon_not_medium_mask = ak.fill_none(loose_muon_not_medium_mask, False)
        loose_muon_not_medium = reco_muon[loose_muon_not_medium_mask]
        

        ######################
        ### Return results ###
        ######################

        def save_array(variable, ID) :
            array = variable[ID]
            array = ak.flatten(array)
            array = ak.to_list(array)
            return array

        #---pt---#

        ele_pt = ak.flatten(ele_pt)
        ele_pt = ak.to_list(ele_pt)
        mu_pt = ak.flatten(mu_pt)
        mu_pt = ak.to_list(mu_pt)
        
        tight_ele_pt = save_array(electron_pt, tight_electron_mask)
        medium_ele_pt = save_array(electron_pt, medium_electron_mask)
        loose_ele_not_tight_pt = save_array(electron_pt, loose_electron_not_tight_mask)
        loose_ele_not_medium_pt = save_array(electron_pt, loose_electron_not_medium_mask)

        tight_muon_pt = save_array(muon_pt, tight_muon_mask)
        medium_muon_pt = save_array(muon_pt, medium_muon_mask)
        loose_muon_not_tight_pt = save_array(muon_pt, loose_muon_not_tight_mask)
        loose_muon_not_medium_pt = save_array(muon_pt, loose_muon_not_medium_mask)

        #---iso---#

        tight_ele_iso = save_array(electron_iso, tight_electron_mask)
        medium_ele_iso = save_array(electron_iso, medium_electron_mask)
        loose_ele_not_tight_iso = save_array(electron_iso, loose_electron_not_tight_mask)
        loose_ele_not_medium_iso = save_array(electron_iso, loose_electron_not_medium_mask)

        tight_muon_iso = save_array(muon_iso, tight_muon_mask)
        medium_muon_iso = save_array(muon_iso, medium_muon_mask)
        loose_muon_not_tight_iso = save_array(muon_iso, loose_muon_not_tight_mask)
        loose_muon_not_medium_iso = save_array(muon_iso, loose_muon_not_medium_mask)


        return {
            "ele_pt": ele_pt,
            "mu_pt": mu_pt,
            "lhe_ele_pt": lhe_ele_pt,
            "lhe_mu_pt":  lhe_mu_pt,
            "tight_ele_pt": tight_ele_pt,
            "medium_ele_pt": medium_ele_pt,
            "loose_ele_not_tight_pt": loose_ele_not_tight_pt,
            "loose_ele_not_medium_pt": loose_ele_not_medium_pt,
            "tight_muon_pt": tight_muon_pt,
            "medium_muon_pt": medium_muon_pt,
            "loose_muon_not_tight_pt": loose_muon_not_tight_pt,
            "loose_muon_not_medium_pt": loose_muon_not_medium_pt,
            "tight_ele_iso": tight_ele_iso,
            "medium_ele_iso": medium_ele_iso,
            "loose_ele_not_tight_iso": loose_ele_not_tight_iso,
            "loose_ele_not_medium_iso": loose_ele_not_medium_iso,
            "tight_muon_iso": tight_muon_iso,
            "medium_muon_iso": medium_muon_iso,
            "loose_muon_not_tight_iso": loose_muon_not_tight_iso,
            "loose_muon_not_medium_iso": loose_muon_not_medium_iso
        }

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--mass', default = '2000')
    parser.add_argument('--debug', default = 'nanoaod-')
    config = parser.parse_args()
    
    files = sorted(glob.glob("/eos/cms/store/group/phys_b2g/ExYukawa/ntu_prod/cgbh_H_M{mass}_rhott06_rhotc04_rhotu00_2017UL/{debug}*".format(mass=config.mass,debug=config.debug)))
    fileset  = {"dataset": files}

    run = processor.Runner(
        executor = processor.FuturesExecutor(workers=16, compression=None),
        schema = NanoAODSchema,
        chunksize = 50000,
        maxchunks = None,
    )

    output = run(
        fileset,
        "Events",
        processor_instance=MySelectionProcessor(),
    )


    ############
    ### Plot ###
    ############
    
    
    def plot (variable, lepton):

        array = [
            (f"tight_{lepton}_{variable}", output[f'tight_{lepton}_{variable}'], "blue"),
            (f"medium_{lepton}_{variable}", output[f'medium_{lepton}_{variable}'], "green"),
            (f"loose_{lepton}_not_tight_{variable}", output[f'loose_{lepton}_not_tight_{variable}'], "orange"),
            (f"loose_{lepton}_not_medium_{variable}", output[f'loose_{lepton}_not_medium_{variable}'], "red"),
        ]

        plt.figure(figsize=(8, 6))
        for name, arr, color in array:
            plt.hist(arr, bins=40, range=(0, 1000), histtype='step', color=color, linewidth=1.5, label=name.replace("_", " "))
        plt.xlabel(f"{lepton} {variable} [GeV]")
        plt.ylabel("Events")
        plt.title(f"{lepton} {variable} Distributions for H+ {config.mass}GeV")
        plt.xticks(np.arange(0, 1025, 50))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"/eos/user/h/hyeh/preselection_result/plot/{lepton}_{variable}_{config.mass}GeV_matching.png", dpi=300)
        plt.close()
        return
    
    variables = ["pt", "iso"]
    leptons = ["ele", "muon"]

    for var in variables:
        for lep in leptons:
            plot(var, lep)



    '''
    pt_bins = np.linspace(0, 1000, 21)
    bin_centers = 0.5 * (pt_bins[1:] + pt_bins[:-1])

    # --------- Electron ---------
    ele_pt = np.asarray(output["ele_pt"])
    den_e, _ = np.histogram(ele_pt, bins=pt_bins)

    tight_ele_pt = np.asarray(output["tight_ele_pt"])
    medium_ele_pt = np.asarray(output["medium_ele_pt"])
    loose_ele_not_tight_pt =  np.asarray(output["loose_ele_not_tight_pt"])
    loose_ele_not_medium_pt =  np.asarray(output["loose_ele_not_medium_pt"])
  
    num_e_tight,  _ = np.histogram(tight_ele_pt,  bins=pt_bins)
    num_e_medium, _ = np.histogram(medium_ele_pt, bins=pt_bins)
    num_e_loose_not_tight,  _ = np.histogram(loose_ele_not_tight_pt,  bins=pt_bins)
    num_e_loose_not_medium,  _ = np.histogram(loose_ele_not_medium_pt,  bins=pt_bins)

    def binom_eff_err(num, den):
        eff = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den>0)
        err = np.zeros_like(eff)
        #eff = np.clip(eff, 0, 2)
        mask = (den > 0) & (num > 0)
        err[mask] = eff[mask] * np.sqrt(1.0 / num[mask] + 1.0 / den[mask])
        #assert np.all(num <= den), "Numerator larger than denominator in some bins!"

        return eff, err
    

    eff_e_tight,  err_e_tight  = binom_eff_err(num_e_tight,  den_e)
    eff_e_medium, err_e_medium = binom_eff_err(num_e_medium, den_e)
    eff_e_loose_not_tight,  err_e_loose_not_tight  = binom_eff_err(num_e_loose_not_tight,  den_e)
    eff_e_loose_not_medium,  err_e_loose_not_medium  = binom_eff_err(num_e_loose_not_medium,  den_e)

    plt.figure(figsize=(8,6))
    plt.errorbar(bin_centers, eff_e_loose_not_tight,  yerr=err_e_loose_not_tight,  fmt='o-', label='Electron loose not tight')
    plt.errorbar(bin_centers, eff_e_loose_not_medium,  yerr=err_e_loose_not_medium,  fmt='o-', label='Electron loose not medium')
    plt.errorbar(bin_centers, eff_e_medium, yerr=err_e_medium, fmt='s-', label='Electron medium (WP90)')
    plt.errorbar(bin_centers, eff_e_tight,  yerr=err_e_tight,  fmt='^-', label='Electron tight (WP80)')
    plt.xlabel(r'Electron $p_T$ [GeV]')
    plt.ylabel('Efficiency')
    #plt.ylim(0, 2)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/eos/user/h/hyeh/preselection_result/plot/eff_vs_pt_electron_{mass}GeV.png".format(mass=config.mass), dpi=300)
    plt.close()

    # --------- Muon ---------
    mu_pt = np.asarray(output["mu_pt"])
    den_mu, _ = np.histogram(mu_pt, bins=pt_bins)

    tight_mu_pt  = np.asarray(output["tight_muon_pt"])
    medium_mu_pt = np.asarray(output["medium_muon_pt"])
    loose_mu_not_tight_pt =  np.asarray(output["loose_muon_not_tight_pt"])
    loose_mu_not_medium_pt =  np.asarray(output["loose_muon_not_medium_pt"])

    num_mu_tight,  _ = np.histogram(tight_mu_pt,  bins=pt_bins)
    num_mu_medium, _ = np.histogram(medium_mu_pt, bins=pt_bins)
    num_mu_loose_not_tight,  _ = np.histogram(loose_mu_not_tight_pt,  bins=pt_bins)
    num_mu_loose_not_medium,  _ = np.histogram(loose_mu_not_medium_pt,  bins=pt_bins)

    eff_mu_tight,  err_mu_tight  = binom_eff_err(num_mu_tight,  den_mu)
    eff_mu_medium, err_mu_medium = binom_eff_err(num_mu_medium, den_mu)
    eff_mu_loose_not_tight,  err_mu_loose_not_tight  = binom_eff_err(num_mu_loose_not_tight,  den_mu)
    eff_mu_loose_not_medium,  err_mu_loose_not_medium  = binom_eff_err(num_mu_loose_not_medium,  den_mu)

    plt.figure(figsize=(8,6))
    plt.errorbar(bin_centers, eff_mu_loose_not_tight,  yerr=err_mu_loose_not_tight,  fmt='o-', label='Muon loose not tight')
    plt.errorbar(bin_centers, eff_mu_loose_not_medium,  yerr=err_mu_loose_not_medium,  fmt='o-', label='Muon loose not medium')
    plt.errorbar(bin_centers, eff_mu_medium, yerr=err_mu_medium, fmt='s-', label='Muon medium')
    plt.errorbar(bin_centers, eff_mu_tight,  yerr=err_mu_tight,  fmt='^-', label='Muon tight')
    plt.xlabel(r'Muon $p_T$ [GeV]')
    plt.ylabel('Efficiency')
    #plt.ylim(0, 2)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/eos/user/h/hyeh/preselection_result/plot/eff_vs_pt_muon_{mass}GeV.png".format(mass=config.mass), dpi=300)
    plt.close()
    '''


    bins = np.linspace(0, 1000, 51)  
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    tight_ele_pt  = output["tight_ele_pt"]
    medium_ele_pt = output["medium_ele_pt"]
    loose_ele_not_tight_pt =  output["loose_ele_not_tight_pt"]
    loose_ele_not_medium_pt =  output["loose_ele_not_medium_pt"]

    tight_mu_pt  = output["tight_muon_pt"]
    medium_mu_pt = output["medium_muon_pt"]
    loose_mu_not_tight_pt =  output["loose_muon_not_tight_pt"]
    loose_mu_not_medium_pt =  output["loose_muon_not_medium_pt"]

    def calculate_eff(pt_array) :
        D = len(pt_array)
        num = np.array([(pt_array >= t).sum() for t in bin_centers], dtype=float)
        eff = num / D
        err = np.sqrt(eff * (1.0 - eff) / D)
        return eff, err
    
    eff_ele_tight, err_ele_tight = calculate_eff(tight_ele_pt)
    eff_ele_medium,  err_ele_medium = calculate_eff(medium_ele_pt)
    eff_ele_loose_not_tight, err_ele_loose_not_tight = calculate_eff(loose_ele_not_tight_pt)
    eff_ele_loose_not_medium,  err_ele_loose_not_medium = calculate_eff(loose_ele_not_medium_pt)
    
    eff_mu_tight, err_mu_tight = calculate_eff(tight_mu_pt)
    eff_mu_medium,  err_mu_medium = calculate_eff(medium_mu_pt)
    eff_mu_loose_not_tight, err_mu_loose_not_tight = calculate_eff(loose_mu_not_tight_pt)
    eff_mu_loose_not_medium,  err_mu_loose_not_medium = calculate_eff(loose_mu_not_medium_pt)

    plt.figure(figsize=(8,6))
    plt.errorbar(bin_centers, eff_ele_tight,  yerr=err_ele_tight,  fmt='^-', label='Muon tight')
    plt.errorbar(bin_centers, eff_ele_medium, yerr=err_ele_medium, fmt='s-', label='Muon medium')
    plt.errorbar(bin_centers, eff_ele_loose_not_tight,  yerr=err_ele_loose_not_tight,  fmt='o-', label='Muon loose not tight')
    plt.errorbar(bin_centers, eff_ele_loose_not_medium,  yerr=err_ele_loose_not_medium,  fmt='o-', label='Muon loose not medium')
    plt.xlabel(r'Electron $p_T$ [GeV]')
    plt.ylabel('Efficiency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/eos/user/h/hyeh/preselection_result/plot/eff_vs_pt_electron_{mass}GeV.png".format(mass=config.mass), dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.errorbar(bin_centers, eff_mu_tight,  yerr=err_mu_tight,  fmt='^-', label='Muon tight')
    plt.errorbar(bin_centers, eff_mu_medium, yerr=err_mu_medium, fmt='s-', label='Muon medium')
    plt.errorbar(bin_centers, eff_mu_loose_not_tight,  yerr=err_mu_loose_not_tight,  fmt='o-', label='Muon loose not tight')
    plt.errorbar(bin_centers, eff_mu_loose_not_medium,  yerr=err_mu_loose_not_medium,  fmt='o-', label='Muon loose not medium')
    plt.xlabel(r'Muon $p_T$ [GeV]')
    plt.ylabel('Efficiency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/eos/user/h/hyeh/preselection_result/plot/eff_vs_pt_muon_{mass}GeV.png".format(mass=config.mass), dpi=300)
    plt.close()

    
