import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import run_uproot_job, futures_executor
import glob


class MySelectionProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):

        ########################
        ### Events selection ###
        ########################

        Electron = events.Electron
        Muon = events.Muon
        LHEPart = events.LHEPart

        lepton = ak.concatenate([Electron, Muon], axis=1)
        nlepton = ak.num(lepton)
        events_1l = events[nlepton == 1]

        Electron_1l = events_1l.Electron
        Muon_1l = events_1l.Muon

        electron_pt = Electron_1l.pt
        muon_pt = Muon_1l.pt
        abs_electron_eta = abs(Electron_1l.eta)
        abs_muon_eta = abs(Muon_1l.eta)

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

        lhe_lepton = events[mask_lhe_1l]
        lhe_electron = events[mask_e_only]
        lhe_muon = events[mask_mu_only]

        #############################
        ### Electron WP selection ###
        #############################

        abs_eta = abs(Electron_1l.eta)
        mva_score = Electron_1l.mvaFall17V2noIso

        pass_WPL = (
            ((abs_eta < 0.8) & (mva_score > -0.83)) |
            ((abs_eta >= 0.8) & (abs_eta < 1.479) & (mva_score > -0.77)) |
            ((abs_eta >= 1.479) & (abs_eta < 2.5) & (mva_score > -0.69))
        )

        pass_WP90 = (
            ((abs_eta < 0.8) & (mva_score > -0.48)) |
            ((abs_eta >= 0.8) & (abs_eta < 1.479) & (mva_score > -0.32)) |
            ((abs_eta >= 1.479) & (mva_score > -0.13))
        )

        pass_WP80 = (
            ((abs_eta < 0.8) & (mva_score > 0.20)) |
            ((abs_eta >= 0.8) & (abs_eta < 1.479) & (mva_score > 0.10)) |
            ((abs_eta >= 1.479) & (mva_score > -0.01))
        )

        fail_WP90 = ~pass_WP90
        fail_WP80 = ~pass_WP80

        ########################
        ### Lepton selection ###
        ########################
        

        tight_electron = Electron_1l[
            ((abs_electron_eta < 1.4442) | ((1.566 < abs_electron_eta) & (abs_electron_eta < 2.5)))
            & pass_WP80
            #& (Electron_1l.miniPFRelIso_all < 0.1)
            & (has_e == True)
        ]

        medium_electron = Electron_1l[
            ((abs_electron_eta < 1.4442) | ((1.566 < abs_electron_eta) & (abs_electron_eta < 2.5)))
            & pass_WP90
            #& (Electron_1l.miniPFRelIso_all < 0.1)
            & (has_e == True)
        ]

        loose_electron_not_tight = Electron_1l[            
            ((abs_electron_eta < 1.4442) | ((1.566 < abs_electron_eta) & (abs_electron_eta < 2.5)))
            #& (Electron_1l.pt > 35)
            & pass_WPL
            & fail_WP80
            #& (Electron_1l.miniPFRelIso_all < 0.4)
            & (has_e == True)
        ]

        loose_electron_not_medium = Electron_1l[            
            ((abs_electron_eta < 1.4442) | ((1.566 < abs_electron_eta) & (abs_electron_eta < 2.5)))
            #& (Electron_1l.pt > 35)
            & pass_WPL
            & fail_WP90
            #& (Electron_1l.miniPFRelIso_all < 0.4)
            & (has_e == True)
        ]


        tight_muon = Muon_1l[            
            (abs_muon_eta < 2.4)
            & (Muon_1l.tightId)
            #& (Muon_1l.miniPFRelIso_all < 0.1)
            & (has_mu == True)
        ]

        medium_muon = Muon_1l[
            (abs_muon_eta < 2.4)
            & (Muon_1l.mediumId)
            #& (Muon_1l.miniPFRelIso_all < 0.1)
            & (has_mu == True)
        ]

        loose_muon_not_tight = Muon_1l[            
            (abs_muon_eta < 2.4)
            #& (Muon_1l.pt > 35)
            & (Muon_1l.looseId)
            & (~Muon_1l.tightId)
            #& (Muon_1l.miniPFRelIso_all < 0.4)
            & (has_mu == True)
        ]

        loose_muon_not_medium = Muon_1l[            
            (abs_muon_eta < 2.4)
            #& (Muon_1l.pt > 35)
            & (Muon_1l.looseId)
            & (~Muon_1l.mediumId)
            #& (Muon_1l.miniPFRelIso_all < 0.4)
            & (has_mu == True)
        ]


        ######################
        ### Return results ###
        ######################

        n_total = len(events)
        n_lhe_1l = len(lhe_lepton)
        n_lhe_ele = len(lhe_electron)
        n_lhe_mu = len(lhe_muon)

        n_tight_ele = ak.sum(ak.num(tight_electron) == 1)
        n_medium_ele = ak.sum(ak.num(medium_electron) == 1)
        n_loose_ele_not_tight = ak.sum(ak.num(loose_electron_not_tight) == 1)
        n_loose_ele_not_medium = ak.sum(ak.num(loose_electron_not_medium) == 1)

        n_tight_mu = ak.sum(ak.num(tight_muon) == 1)
        n_medium_mu = ak.sum(ak.num(medium_muon) == 1)
        n_loose_mu_not_tight = ak.sum(ak.num(loose_muon_not_tight) == 1)
        n_loose_mu_not_medium = ak.sum(ak.num(loose_muon_not_medium) == 1)


        return {
            "n_total": n_total,
            "n_lhe_1l": n_lhe_1l,
            "n_lhe_ele": n_lhe_ele,
            "n_lhe_mu": n_lhe_mu,
            "n_tight_ele": n_tight_ele,
            "n_medium_ele": n_medium_ele,
            "n_loose_ele_not_tight": n_loose_ele_not_tight,
            "n_loose_ele_not_medium": n_loose_ele_not_medium,
            "n_tight_mu": n_tight_mu,
            "n_medium_mu": n_medium_mu,
            "n_loose_mu_not_tight": n_loose_mu_not_tight,
            "n_loose_mu_not_medium": n_loose_mu_not_medium,
        }

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":
    
    files = sorted(glob.glob("/eos/cms/store/group/phys_b2g/ExYukawa/ntu_prod/cgbh_H_M3000_rhott06_rhotc04_rhotu00_2017UL/nanoaod-*"))
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
    

    ##############
    ### Output ###
    ##############

    n_lhe_ele = output['n_lhe_ele']
    n_lhe_mu = output['n_lhe_mu']
    
    n_tight_ele = output['n_tight_ele']
    n_medium_ele = output['n_medium_ele']
    n_loose_ele_not_tight = output['n_loose_ele_not_tight']
    n_loose_ele_not_medium = output['n_loose_ele_not_medium']

    n_tight_mu = output['n_tight_mu']
    n_medium_mu = output['n_medium_mu']
    n_loose_mu_not_tight = output['n_loose_mu_not_tight']
    n_loose_mu_not_medium = output['n_loose_mu_not_medium']

    print("==================== Summary =======================")
    print(f"Total number of events            : {output['n_total']}")
    print("----------------------------------------------------")
    print(f"LHE level in reco level 1l region : {output['n_lhe_1l']}")
    print(f"  ├─ Electron only                : {output['n_lhe_ele']}")
    print(f"  └─ Muon only                    : {output['n_lhe_mu']}")
    print("----------------------------------------------------")
    print(f"Reco level tight electron events            : {output['n_tight_ele']}")
    print(f"Reco level medium electron events           : {output['n_medium_ele']}")
    print(f"Reco level loose electron not tight events  : {output['n_loose_ele_not_tight']}")
    print(f"Reco level loose electron not medium events : {output['n_loose_ele_not_medium']}")
    print(f"Reco level tight muon events                : {output['n_tight_mu']}")
    print(f"Reco level medium muon events               : {output['n_medium_mu']}")
    print(f"Reco level loose muon not tight events      : {output['n_loose_mu_not_tight']}")
    print(f"Reco level loose muon not medium events     : {output['n_loose_mu_not_medium']}")
    print("==================== Efficiency ====================")
    print("tight electron                :", n_tight_ele/n_lhe_ele *100, "%")
    print("medium electron               :", n_medium_ele/n_lhe_ele *100, "%")
    print("loose electron but not tight  :", n_loose_ele_not_tight/n_lhe_ele *100, "%")
    print("loose electron but not medium :", n_loose_ele_not_medium/n_lhe_ele *100, "%")
    print("tight muon                    :", n_tight_mu/n_lhe_mu *100, "%")
    print("medium muon                   :", n_medium_mu/n_lhe_mu *100, "%")
    print("loose muon but not tight      :", n_loose_mu_not_tight/n_lhe_mu *100, "%")
    print("loose muon but not medium     :", n_loose_mu_not_medium/n_lhe_mu *100, "%")
    print("----------------------------------------------------")
    