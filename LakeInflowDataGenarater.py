import pcraster
from pcraster import *
from pcraster.framework import *
from osgeo import gdal, ogr
import numpy as np
import tqdm, os
import pandas as pd


class RunoffModelDataManager:
    """
    All file in tif format with WGS 1984-Mercator projection, resolution of pixel is 1000 m.
        watershedMask: with value mask of watershed according to LakeID int
        Pr: format as 'Year.tif', read according to the watershedMask
        DEM: read according to the watershedMask
        LUCC: land use in year, with a dic {'LUCCYear.tif': [year1, year2...]} points represent which year
        LUCC lookup table: for set water speed
        SoilType: soil type for identify the Soil Maximum Water Holding Capacity (SMWHC)
        ST lookup table: for set SMWHC
    For every lake, has those data:
        LDDMap: ldd of the mask watershed, created by DEM
        soilMoistureMaxMap: unit in mm, created by SoilType and ST lookup table
        waterSpeedMap: unit in m/day (according to pixel resolution), created by DEM and LUCC (may exclude)
        PrMapDic: daily pr data (mm/d), with date index
        EvaMapDic: daily eva data (mm/d), with date index
    """

    def __init__(self, **kwargs):
        yearBound = kwargs['yearBound']
        self.dateRange = pd.date_range(start='{}-1-1'.format(yearBound[0]), end='{}-12-31'.format(yearBound[-1])).strftime('%Y-%m-%d').tolist()
        self.YearRange = pd.date_range(start='{}-1-1'.format(yearBound[0]), end='{}-12-31'.format(yearBound[-1]), freq='Y').strftime('%Y').tolist()
        self.lakeIDs = kwargs['lakeIDs']

        self.WatershedShpDataset = ogr.Open(kwargs['watershedShpPath'])
        self.WatershedShpFC = self.WatershedShpDataset.GetLayer()

        MaskTifDataset = gdal.Open(kwargs['maskTifPath'])
        self.fullMapGeoTransform = MaskTifDataset.GetGeoTransform()
        self.projection = MaskTifDataset.GetProjection()

        self.DEMtifDataset = gdal.Open(kwargs['DEMtifPath'])
        self.DEMtifBand1 = self.DEMtifDataset.GetRasterBand(1)
        self.DEMNodata = self.DEMtifBand1.GetNoDataValue()

        self.PrTifDir = kwargs['PrTifDir']
        self.EvaTifDir = kwargs['EvaTifDir']

        self.idxFieldName = 'LakeID'

        self.CloneParasDic = {}
        self.MaskArrayDic = {}
        self.PrArraysDic = {}
        self.EvaArraysDic = {}
        self.watershedBoundGeoTransform_Dic = {}
        self.watershedBoundIndexOffsets_Dic = {}
        self.ws = 0.1 if 'ws' not in kwargs.keys() else kwargs['ws'] # water flow speed (m/s)
        self.secondInStep = 86400 if 'secondInStep' not in kwargs.keys() else kwargs['secondInStep']
        self.soilMoistureMax = 25. if 'soilMoistureMax' not in kwargs.keys() else kwargs['soilMoistureMax']
        self.iniSoilMoisture = 15. if 'iniSoilMoisture' not in kwargs.keys() else kwargs['iniSoilMoisture']
        self.PrZoomOut = 10 if 'PrZoomOut' not in kwargs.keys() else kwargs['PrZoomOut']
        self.EvaZoomOut = 10 if 'EvaZoomOut' not in kwargs.keys() else kwargs['EvaZoomOut']
        self.noDataValue = -9999 if 'noDataValue' not in kwargs.keys() else kwargs['noDataValue']

        self.readWatershed(saveDir=kwargs['saveDir'])
        self.readPrEvaDemLddMaps(saveDir=kwargs['saveDir'])

    def readPrEvaDemLddMaps(self, saveDir):
        for year in self.YearRange:
            yearPrDataset = gdal.Open(os.path.join(self.PrTifDir, '{}-01-01_{}-01-01total_precipitation_sum.tif'.format(year, int(year) + 1)))
            yearEvaDataset = gdal.Open(os.path.join(self.EvaTifDir, '{}-01-01_{}-01-01total_evaporation_sum.tif'.format(year, int(year) + 1)))
            pbar = tqdm.tqdm(total=365, ncols=150, desc='Load {} Pr/Eva Data'.format(year))
            for i, date in enumerate(pd.date_range(start='{}-1-1'.format(year), end='{}-12-31'.format(year)).strftime('%Y-%m-%d').tolist()):
                PrDateBand = yearPrDataset.GetRasterBand(i + 1)
                EvaDateBand = yearEvaDataset.GetRasterBand(i + 1)
                for key in self.CloneParasDic.keys():
                    maskArray = self.MaskArrayDic[key]
                    watershedBoundIndexOffsets = self.watershedBoundIndexOffsets_Dic[key]

                    PrDateArray = PrDateBand.ReadAsArray(int(watershedBoundIndexOffsets[2] / self.PrZoomOut), int(watershedBoundIndexOffsets[0] / self.PrZoomOut),
                                                            int((watershedBoundIndexOffsets[3] - watershedBoundIndexOffsets[2]) / self.PrZoomOut)+1,
                                                            int((watershedBoundIndexOffsets[1] - watershedBoundIndexOffsets[0]) / self.PrZoomOut)+1)
                    EvaDateArray = EvaDateBand.ReadAsArray(int(watershedBoundIndexOffsets[2] / self.EvaZoomOut), int(watershedBoundIndexOffsets[0] / self.EvaZoomOut),
                                                            int((watershedBoundIndexOffsets[3] - watershedBoundIndexOffsets[2]) / self.EvaZoomOut) + 1,
                                                            int((watershedBoundIndexOffsets[1] - watershedBoundIndexOffsets[0]) / self.EvaZoomOut) + 1)

                    PrDateArray = np.where(PrDateArray == PrDateBand.GetNoDataValue(), 0, PrDateArray)
                    prMean = np.nanmean(PrDateArray) if not np.isnan(np.nanmean(PrDateArray)) else 0.1
                    PrDateArray = np.nan_to_num(PrDateArray, nan=prMean)

                    EvaDateArray = np.where(EvaDateArray == EvaDateBand.GetNoDataValue(), 0, EvaDateArray)
                    evaMean = np.nanmean(EvaDateArray) if not np.isnan(np.nanmean(EvaDateArray)) else 0.1
                    EvaDateArray = np.nan_to_num(EvaDateArray, nan=evaMean)

                    PrDateArray = np.repeat(np.repeat(PrDateArray, self.PrZoomOut, axis=0), self.PrZoomOut, axis=1)
                    EvaDateArray = np.repeat(np.repeat(EvaDateArray, self.EvaZoomOut, axis=0), self.EvaZoomOut, axis=1)

                    PrDateArray = self.checkArrayShape(maskArray, PrDateArray)
                    EvaDateArray = self.checkArrayShape(maskArray, EvaDateArray)

                    PrDateMaskArray = np.where(maskArray, PrDateArray, self.noDataValue)
                    EvaDateMaskArray = np.where(maskArray, EvaDateArray, self.noDataValue)
                    self.PrArraysDic[key][date] = PrDateMaskArray
                    self.EvaArraysDic[key][date] = EvaDateMaskArray
                pbar.update()
            pbar.close()
            pbar = tqdm.tqdm(total=len(self.CloneParasDic), ncols=150, desc='Write {} Pr/Eva Data'.format(year))
            for key, PrMapDateDic in self.PrArraysDic.items():
                setclone(*self.CloneParasDic[key])
                for date, prArray, evaArray in zip(PrMapDateDic.keys(), PrMapDateDic.values(), self.EvaArraysDic[key].values()):
                    PrMap = numpy2pcr(pcr.Scalar, prArray, self.noDataValue)
                    EvaMap = numpy2pcr(pcr.Scalar, evaArray, self.noDataValue)
                    self.writeMapData(os.path.join(saveDir, 'PRMAP'), 'PR-{}-{}'.format(key, date), PrMap, key)
                    self.writeMapData(os.path.join(saveDir, 'EVAMAP'), 'EVA-{}-{}'.format(key, date), EvaMap, key)
                pbar.update()
                self.PrArraysDic[key] = {}
                self.EvaArraysDic[key] = {}
            pbar.close()

    def checkArrayShape(self, standardArray, clipArray):
        if standardArray.shape == clipArray.shape:
            return clipArray
        else:
            standardRows, standardCols = standardArray.shape
            clipRows, clipCols = clipArray.shape
            if standardRows > clipRows and standardCols > clipCols:
                rowRepeatLs, colRepeatLs = [1] * (clipRows - 1), [1] * (clipCols - 1)
                rowRepeatLs.append(standardRows - clipRows + 1)
                colRepeatLs.append(standardCols - clipCols + 1)
                newArray = np.repeat(clipArray, rowRepeatLs, axis=0)
                newArray = np.repeat(newArray, colRepeatLs, axis=1)
            elif standardRows >= clipRows and standardCols < clipCols:
                newArray = np.zeros_like(standardArray, dtype=np.float32)
                newArray[: clipRows, :] = clipArray[:, : standardCols]
                rowRepeatLs = [1] * (clipRows - 1)
                rowRepeatLs.append(standardRows - clipRows + 1)
                newArray = np.repeat(newArray, rowRepeatLs, axis=0)
            elif standardRows < clipRows and standardCols >= clipCols:
                newArray = np.zeros_like(standardArray, dtype=np.float32)
                newArray[:, : clipCols] = clipArray[: standardRows, :]
                colRepeatLs = [1] * (clipCols - 1)
                colRepeatLs.append(standardCols - clipCols + 1)
                newArray = np.repeat(newArray, colRepeatLs, axis=1)
            else:   # standardRows <= clipRows and standardCols <= clipCols
                newArray = np.zeros_like(standardArray, dtype=np.float32)
                newArray[:, :] = clipArray[: standardRows, : standardCols]
            return newArray

    def readWatershed(self, saveDir):
        """
        DO: read every watershed shape, and rasterize them to create mask map
        SAVE:
            mask map -> clone map
            watershed bounds geo. transform -> position identity
            index offset bounds -> clip pr/dem/eva array
        """
        shp_driver = ogr.GetDriverByName("Memory")
        img_driver = gdal.GetDriverByName("MEM")
        feature = self.WatershedShpFC.GetNextFeature()
        featureCount = self.WatershedShpFC.GetFeatureCount()
        pbar = tqdm.tqdm(total=len(self.lakeIDs), ncols=150, desc='Load Watershed Data (MASK/DEM/LDD/WS/SMM)')
        while feature:
            indexFieldValue = str(feature.GetField(self.idxFieldName))
            if indexFieldValue not in self.lakeIDs:
                feature = self.WatershedShpFC.GetNextFeature()
                continue
            geo = feature.GetGeometryRef()
            geo_bounds = geo.GetEnvelope()
            # create temp shape dataset
            temp_shp_dataset = shp_driver.CreateDataSource("temp")
            if geo.GetGeometryType() == ogr.wkbPolygon:
                temp_shp = temp_shp_dataset.CreateLayer("temp", self.WatershedShpFC.GetSpatialRef(), ogr.wkbPolygon)
                temp_shp.CreateFeature(feature.Clone())
            elif geo.GetGeometryType() == ogr.wkbMultiPolygon:
                temp_shp = temp_shp_dataset.CreateLayer("temp", self.WatershedShpFC.GetSpatialRef(), ogr.wkbMultiPolygon)
                temp_shp.CreateFeature(feature.Clone())
            else:
                raise LookupError('The geo. type is not Polygon or MultiPolygon, but invalid {}'.format(
                    str(ogr.GeometryTypeToName(geo.GetGeometryType()))))
            # get the offsets of shape bounds in the dem
            watershedBound_indexOffsets = [
                int((geo_bounds[3] - self.fullMapGeoTransform[3]) / self.fullMapGeoTransform[5]),
                int((geo_bounds[2] - self.fullMapGeoTransform[3]) / self.fullMapGeoTransform[5]) + 1,
                int((geo_bounds[0] - self.fullMapGeoTransform[0]) / self.fullMapGeoTransform[1]),
                int((geo_bounds[1] - self.fullMapGeoTransform[0]) / self.fullMapGeoTransform[1]) + 1,
            ]
            # create a temp tif contain the geo for rasterize
            watershedBound_geoTransform = [
                self.fullMapGeoTransform[0] + (watershedBound_indexOffsets[2] * self.fullMapGeoTransform[1]),
                self.fullMapGeoTransform[1], 0.0,
                self.fullMapGeoTransform[3] + (watershedBound_indexOffsets[0] * self.fullMapGeoTransform[5]), 0.0,
                self.fullMapGeoTransform[5]
            ]
            temp_img_dataset = img_driver.Create("",
                                                 (watershedBound_indexOffsets[3] - watershedBound_indexOffsets[2]),
                                                 (watershedBound_indexOffsets[1] - watershedBound_indexOffsets[0]),
                                                 1,
                                                 gdal.GDT_Byte)
            temp_img_dataset.SetGeoTransform(watershedBound_geoTransform)

            # trans the temp shape to tiff and write to the temp tif band 1 with constant 1, others set as noDataValue
            gdal.RasterizeLayer(temp_img_dataset, [1], temp_shp, burn_values=[1])
            maskArray = temp_img_dataset.ReadAsArray()
            # maskArray = np.where(maskArray == 0, self.noDataValue, maskArray)
            self.MaskArrayDic[indexFieldValue] = maskArray.astype(np.bool_)

            # Create a PCRaster RasterSpace from coordinate, trans array to pcr
            setclone(maskArray.shape[0], maskArray.shape[1], watershedBound_geoTransform[1], watershedBound_geoTransform[0], watershedBound_geoTransform[3])
            maskMap = numpy2pcr(pcr.Scalar, maskArray, 0)
            self.writeMapData(saveDir=os.path.join(saveDir, 'MASKMap'), filename='MASK-{}'.format(indexFieldValue), map=maskMap, key=indexFieldValue)

            self.CloneParasDic[indexFieldValue] = [maskArray.shape[0], maskArray.shape[1], watershedBound_geoTransform[1], watershedBound_geoTransform[0], watershedBound_geoTransform[3]]
            self.watershedBoundGeoTransform_Dic[indexFieldValue] = watershedBound_geoTransform
            self.watershedBoundIndexOffsets_Dic[indexFieldValue] = watershedBound_indexOffsets

            DEMDataArray = self.DEMtifBand1.ReadAsArray(watershedBound_indexOffsets[2], watershedBound_indexOffsets[0],
                                                        watershedBound_indexOffsets[3] - watershedBound_indexOffsets[2],
                                                        watershedBound_indexOffsets[1] - watershedBound_indexOffsets[0])
            DEMDataArray = np.where(DEMDataArray == self.DEMNodata, 0, DEMDataArray)
            DEMDataArray = self.checkArrayShape(maskArray, DEMDataArray)
            DEMDateMaskArray = np.where(self.MaskArrayDic[indexFieldValue], DEMDataArray, self.noDataValue)
            DEMMap = numpy2pcr(pcr.Scalar, DEMDateMaskArray, self.noDataValue)
            self.writeMapData(saveDir=os.path.join(saveDir, 'DEMMap'), filename='DEM-{}'.format(indexFieldValue), map=DEMMap, key=indexFieldValue)

            LDDMap = lddcreate(DEMMap, 1e31, 1e31, 1e31, 1e31)
            self.writeMapData(saveDir=os.path.join(saveDir, 'LDDMap'), filename='LDD-{}'.format(indexFieldValue), map=LDDMap, key=indexFieldValue)

            waterSpeedArray = np.ones_like(maskArray, dtype=np.float32) * self.ws * self.secondInStep
            waterSpeedMaskArray = np.where(self.MaskArrayDic[indexFieldValue], waterSpeedArray, self.noDataValue)
            WSMap = numpy2pcr(pcr.Scalar, waterSpeedMaskArray, self.noDataValue)
            self.writeMapData(saveDir=os.path.join(saveDir, 'WSMap'), filename='WS-{}'.format(indexFieldValue), map=WSMap, key=indexFieldValue)

            soilMoistureArray = np.ones_like(maskArray, dtype=np.float32) * self.soilMoistureMax
            soilMoistureMaskArray = np.where(self.MaskArrayDic[indexFieldValue], soilMoistureArray, self.noDataValue)
            SMMMap = numpy2pcr(pcr.Scalar, soilMoistureMaskArray, self.noDataValue)
            self.writeMapData(saveDir=os.path.join(saveDir, 'SMMMap'), filename='SMM-{}'.format(indexFieldValue), map=SMMMap, key=indexFieldValue)

            iniSoilMoistureArray = np.ones_like(maskArray, dtype=np.float32) * self.iniSoilMoisture
            iniSoilMoistureMaskArray = np.where(self.MaskArrayDic[indexFieldValue], iniSoilMoistureArray, self.noDataValue)
            ISMMap = numpy2pcr(pcr.Scalar, iniSoilMoistureMaskArray, self.noDataValue)
            self.writeMapData(saveDir=os.path.join(saveDir, 'ISMMap'), filename='ISM-{}'.format(indexFieldValue), map=ISMMap, key=indexFieldValue)

            self.PrArraysDic[indexFieldValue] = {}
            self.EvaArraysDic[indexFieldValue] = {}
            feature = self.WatershedShpFC.GetNextFeature()
            pbar.update()
        pbar.close()

    def writeMapData(self, saveDir, filename, map, key, ifSaveTif=False):
        _ = None if os.path.exists(saveDir) else os.mkdir(saveDir)

        # write map and tif
        report(map, os.path.join(saveDir, '{}.map'.format(filename)))
        if ifSaveTif:
            driver = gdal.GetDriverByName('GTiff')
            array = pcr2numpy(map, self.noDataValue)
            array = np.where(array, array, self.noDataValue)
            maskDataset = driver.Create(os.path.join(saveDir, '{}.tif'.format(filename)), pcraster.clone().nrCols(), pcraster.clone().nrRows(), 1, gdal.GDT_Float32)
            maskDataset.SetGeoTransform(self.watershedBoundGeoTransform_Dic[key])
            maskDataset.SetProjection(self.projection)
            maskDatasetBand1 = maskDataset.GetRasterBand(1)
            maskDatasetBand1.WriteArray(array)
            maskDatasetBand1.SetNoDataValue(self.noDataValue)
            maskDataset = None


if __name__ == '__main__':
    Cases_df = pd.read_excel(r'Data\Model.xlsx', sheet_name='Cases', header=0, index_col='Index')
    AllCases_dic = {lake_id: lake_name for lake_id, lake_name in zip(list(Cases_df.columns), list(Cases_df.loc['Name', :]))}
    UsedCases_dic = {lake_id: lake_name for lake_id, lake_name in zip(list(Cases_df.columns), list(Cases_df.loc['Name', :])) if int(Cases_df.loc['Use', lake_id]) == 1}

    t = RunoffModelDataManager(**{
        'watershedShpPath': r'LakeWatershed_Mercator.shp',
        'maskTifPath': r'LakeIDIntRaster1000_Mercator.tif',
        'DEMtifPath': r'DEM1000_Mercator.tif',
        'PrTifDir': r'2005-2022PrMercator',
        'EvaTifDir': r'2005-2022EvaMercator',
        'LUCCTif': r'LUCC1000_Mercator.tif',
        'yearBound': ['2005', '2022'],
        'lakeIDs': list(UsedCases_dic.keys()),
        'saveDir': r'Data\PCRaster',
    })
