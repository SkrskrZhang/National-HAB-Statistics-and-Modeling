import numpy as np
from pcraster import *
from pcraster.framework import *
import tqdm, os
import pandas as pd


class LakeInflowModel:
    """
    Model the water flow to lake from attachment area.
    Consider:
        - Pr: .map, mm/day
        - Soil moisture : .map, mm, has a critical level as the threshold to transport
        - Water flow speed: .map, m/timestep, according to DEM or ?
        - Eva: as a constant? or calc from Penman eq.
    Process:
        - 1. calc soil moisture during pr and eva, max(Pr + soilMoisture - Eva, 0)
        - 2. get the water could be transport (critical moisture threshold), max(1. - soilMoistureMax, 0)
        - 3. calc the water transport under the water speed, accutraveltimeflux
        - 4. set the next step soil moisture, accutraveltimestate + min(soilMoisture, soilMoistureMax)
    Unit:
        Input raster: 1000 * 1000 m
        Pr/SMM/Eva: mm
    """
    def __init__(self, kwargs):
        yearBound = kwargs['yearBound']
        self.dateRange = pd.date_range(start='{}-1-1'.format(yearBound[0]), end='{}-12-31'.format(yearBound[-1]))
        LakeID = kwargs['LakeID']
        self.LakeID = LakeID

        # Load mask map
        maskDir = kwargs['maskDir']
        self.mask_map = readmap(os.path.join(maskDir, 'MASK-{}.map').format(LakeID))
        setclone(os.path.join(maskDir, 'MASK-{}.map').format(LakeID))

        # Load ini soil m
        ismDir = kwargs['ismDir']
        self.soilMoistureMapTimeVar = readmap(os.path.join(ismDir, 'ISM-{}.map').format(LakeID))

        # Load max soil m
        smmDir = kwargs['smmDir']
        self.soilMaxMoistureMap = readmap(os.path.join(smmDir, 'SMM-{}.map').format(LakeID))

        # Load LDD
        lddDir = kwargs['lddDir']
        self.lddMap = readmap(os.path.join(lddDir, 'LDD-{}.map').format(LakeID))

        # Load water speed
        wsDir = kwargs['wsDir']
        self.wsMap = readmap(os.path.join(wsDir, 'WS-{}.map').format(LakeID))

        # Set save dir
        saveDir = kwargs['saveDir']
        self.saveDir = saveDir
        _ = None if os.path.exists(self.saveDir) else os.mkdir(self.saveDir)
        self.saveFluxDir = os.path.join(saveDir, 'FluxMap')
        _ = None if os.path.exists(self.saveFluxDir) else os.mkdir(self.saveFluxDir)
        self.saveStateDir = os.path.join(saveDir, 'StateMap')
        _ = None if os.path.exists(self.saveStateDir) else os.mkdir(self.saveStateDir)
        self.saveSoilDir = os.path.join(saveDir, 'SoilMap')
        _ = None if os.path.exists(self.saveSoilDir) else os.mkdir(self.saveSoilDir)
        self.saveFigDir = os.path.join(saveDir, 'FlowFigs')
        _ = None if os.path.exists(self.saveFigDir) else os.mkdir(self.saveFigDir)

        # Set other
        self.transportableMap = scalar(0)
        self.unitConvert = 1000     # convert mm*cellArea to m3
        self.prDir = kwargs['prDir']
        self.evaDir = kwargs['evaDir']
        self.FlowDF = pd.DataFrame(index=self.dateRange)

        self.dynamic(showBar=True)

    def dynamic(self, showBar=True):
        if showBar:
            pbar = tqdm.tqdm(total=len(self.dateRange), ncols=150, desc='RUN {}'.format(self.LakeID))
        for i, date in enumerate(self.dateRange):
            dateStr = date.strftime('%Y-%m-%d')

            # 0. load time vars of eva and pr
            d_eva = readmap(os.path.join(self.evaDir, 'EVA-{}-{}.map').format(self.LakeID, dateStr))   # positive means downward flux (negative is eva)
            d_pr = readmap(os.path.join(self.prDir, 'PR-{}-{}.map').format(self.LakeID, dateStr))
            d_totalSmmBegin = float(pcr.maptotal(ifthenelse(self.mask_map > 0, self.soilMoistureMapTimeVar, scalar(0))))

            # 1. calc soil moisture during pr and eva, max(Pr + soilMoisture - Eva, 0), calc true eva sum
            d_soilMoistureMap = self.soilMoistureMapTimeVar + d_pr + d_eva
            self.soilMoistureMapTimeVar = ifthenelse(d_soilMoistureMap > 0, d_soilMoistureMap, scalar(0))
            d_trueEvaMap = d_soilMoistureMap - d_eva - self.soilMoistureMapTimeVar
            d_totalEva = float(pcr.maptotal(ifthenelse(self.mask_map > 0, d_trueEvaMap, scalar(0))))
            d_totalPr = float(pcr.maptotal(ifthenelse(self.mask_map > 0, d_pr, scalar(0))))

            # 2. get the water could be transport (critical moisture threshold), max(1. - soilMoistureMax, 0)
            d_transportableMap = self.soilMoistureMapTimeVar - self.soilMaxMoistureMap
            self.transportableMap = ifthenelse(d_transportableMap > 0, d_transportableMap, scalar(0))

            # 3. calc the water transport under the water speed, accutraveltimeflux
            d_flux = accutraveltimeflux(self.lddMap, self.transportableMap, self.wsMap)

            # 4. set the next step soil moisture, accutraveltimestate + min(soilMoisture, soilMoistureMax)
            d_state = accutraveltimestate(self.lddMap, self.transportableMap, self.wsMap)
            d_nextSoilMoistureMap = d_state + ifthenelse(self.soilMoistureMapTimeVar > self.soilMaxMoistureMap, self.soilMaxMoistureMap, self.soilMoistureMapTimeVar)
            self.soilMoistureMapTimeVar = d_nextSoilMoistureMap
            d_totalSmmEnd = float(pcr.maptotal(ifthenelse(self.mask_map > 0, self.soilMoistureMapTimeVar, scalar(0))))
            d_totalOut = d_totalSmmBegin + d_totalPr + d_totalEva - d_totalSmmEnd       # UNIT: 1000 m3 [1mm * 1000000m2]
            self.FlowDF.loc[date, '{}-InputFlow'.format(self.LakeID)] = float(d_totalOut)
            self.FlowDF.loc[date, '{}-InputFlowABS'.format(self.LakeID)] = float(max(float(d_totalOut), 0))
            self.FlowDF.loc[date, '{}-SmmBegin'.format(self.LakeID)] = float(d_totalSmmBegin)
            self.FlowDF.loc[date, '{}-TotalEva'.format(self.LakeID)] = float(d_totalEva)
            self.FlowDF.loc[date, '{}-TotalPr'.format(self.LakeID)] = float(d_totalPr)
            self.FlowDF.loc[date, '{}-SmmEnd'.format(self.LakeID)] = float(d_totalSmmEnd)
            report(ifthenelse(self.mask_map > 0, d_flux, scalar(0)), os.path.join(self.saveFluxDir, 'Flux-{}-{}.map'.format(self.LakeID, dateStr)))
            report(ifthenelse(self.mask_map > 0, d_state, scalar(0)), os.path.join(self.saveStateDir, 'State-{}-{}.map'.format(self.LakeID, dateStr)))
            report(ifthenelse(self.mask_map > 0, self.soilMoistureMapTimeVar, scalar(0)), os.path.join(self.saveSoilDir, 'SoilMoisture-{}-{}.map'.format(self.LakeID, dateStr)))
            if showBar:
                pbar.update()
            report(d_flux, '{}.map'.format(date.strftime('%Y-%m-%d')))
        if showBar:
            pbar.close()
        self.FlowDF.to_excel(os.path.join(self.saveDir, '{}.xlsx'.format(self.LakeID)), index=True, index_label='Date', sheet_name=self.LakeID)

    @staticmethod
    def mergeFlowData(saveDir):
        df = pd.DataFrame(index=pd.date_range(start='2005-1-1', end='2022-12-31', freq='d'))
        for lakeID in UsedCases_dic.keys():
            lake_df = pd.read_excel(os.path.join(saveDir, '{}.xlsx'.format(lakeID)), header=0)
            lake_df = lake_df.set_index(pd.to_datetime(lake_df['Date']))
            df[lakeID] = lake_df['{}-InputFlowABS'.format(lakeID)]
        df.fillna(1, inplace=True)
        df.to_excel('MergeFlow.xlsx', sheet_name='InflowWL', index=True, index_label='Date')


if __name__ == '__main__':

    Cases_df = pd.read_excel(r'Data\Model.xlsx', sheet_name='Cases', header=0, index_col='Index')
    AllCases_dic = {lake_id: lake_name for lake_id, lake_name in zip(list(Cases_df.columns), list(Cases_df.loc['Name', :]))}
    UsedCases_dic = {lake_id: lake_name for lake_id, lake_name in zip(list(Cases_df.columns), list(Cases_df.loc['Name', :])) if int(Cases_df.loc['Use', lake_id]) == 1}

    kwargsLs = [
        {
            'LakeID': lakeID,
            'saveDir': r'Data\PCRaster\Results',
            'prDir': r'Data\PCRaster\PRMap',
            'evaDir': r'Data\PCRaster\EVAMap',
            'maskDir': r'Data\PCRaster\MASKMap',
            'lddDir': r'Data\PCRaster\LDDMap',
            'smmDir': r'Data\PCRaster\SMMMap',
            'ismDir': r'Data\PCRaster\ISMMap',
            'wsDir': r'Data\PCRaster\WSMap',
            'yearBound': [2005, 2022]
        }
        for lakeID in list(UsedCases_dic.keys())
    ]
    for kwargs in kwargsLs:
        LakeInflowModel(kwargs)
    LakeInflowModel.mergeFlowData(saveDir=r'Data\PCRaster\Results')
