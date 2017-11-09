# A file containing all our models and data processing methods

# import statement and extra libraries used
import numpy as np
import pandas as pd
import os
import csv
import math
import matplotlib.pyplot as plt
import sklearn
import statistics as st
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datetime import datetime, date
from datetime import timedelta





class RmspeObjective:
    hessian = None

    def __call__(self, predicted, target):
        target = target.get_label()
        # I suspect this is necessary since XGBoost is using 32 bit floats
        # and I'm getting some sort of under/overflow, but that's just a guess
        if self.hessian is None:
            scale = target.max()
            valid = (target > 0)
            self.hessian = np.where(valid, 1.0 / (target / scale) ** 2, 0)
        grad = (predicted - target) * self.hessian
        # I suspect (from experiment not from actually reading the relevant paper)
        # that what is important is the ratio of grad to hess.  That's why (I think)
        # I can get away with returning these values, which should be divided by
        # scale**2
        return grad, self.hessian



class xgBoostModel:

    # static variables: Constant used in this homework
    DATE_TIME_FORMAT_DEV = "%d/%m/%Y"
    DATE_TIME_FORMAT_REAL = "%Y-%m-%d"
    DATE_TIME_FORMAT_WEEK = "%Y-W%W-%w"
    MIN_BOOLEAN_INDEX_TRAIN = 5
    MAX_BOOLEAN_INDEX_TRAIN = 8
    RAW_FEATURE_NUMBER_TRAIN = 9
    MIN_BOOLEAN_INDEX_TEST = 4
    MAX_BOOLEAN_INDEX_TEST = 7
    RAW_FEATURE_NUMBER_TEST = 8
    STORE_COMPETITION_SINCE_DEFAULT_TIME = date(2009, 3, 9)
    STORE_NO_PROMOTION_SINCE_CONSTANT_TIME = date(2999, 1, 1)  # we assume this datetime is big enough
    STORE_NO_COMPETITION_SINCE_CONSTANT_TIME = date(2999, 1, 1)  # we assume this datetime is big enough
    STORE_NO_PROMO_INTERVAL_STRING = "No Promotion"
    OPEN_BOOLEAN_INDEX = 2


    def __init__(self, dataFilePath=None, testFilePath=None, StoreInfoPath=None):
        # Raw Data Read
        self.headerRawTrain, self.dataRawTrain = \
            xgBoostModel.dataLoadExtract(dataFilePath, xgBoostModel.MIN_BOOLEAN_INDEX_TRAIN,
                                         xgBoostModel.MAX_BOOLEAN_INDEX_TRAIN, 7)
        self.headerRawTest, self.dataRawTest = \
            xgBoostModel.dataLoadExtract(testFilePath, xgBoostModel.MIN_BOOLEAN_INDEX_TEST,
                                         xgBoostModel.MAX_BOOLEAN_INDEX_TEST, 6)
        self.headerRawStore, self.dataRawStore = xgBoostModel.storeLoadExtract(StoreInfoPath)

    def processStore(self):
        # Store Information and pre-processing
        self.headerStore, self.dataStore = xgBoostModel.storeInfoConverter(self.headerRawStore, self.dataRawStore,
                                                                           self.dataRawTrain)
        # Split training to features and labels
        self.dataTrain, self.headerTrain, self.labelTrain = xgBoostModel.trainDataLabelSplit(self.dataRawTrain,
                                                                                             self.headerRawTrain)

    def naturalJoin(self):
        # Store Training info natural join
        self.headerTrainClean, self.dataTrainClean = xgBoostModel.storeNaturalJoin(self.dataStore, self.dataTrain,
                                                                                   self.headerStore, self.headerTrain)
        self.headerTestClean, self.dataTestClean = xgBoostModel.storeNaturalJoin(self.dataStore, self.dataRawTest,
                                                                                 self.headerStore, self.headerRawTest)

        # Remove zero entries on all (training, testing and labels)
        self.dataTestRealClean = xgBoostModel.removeZeroInDataTestClean(self.dataTestClean,
                                                                        openBooleanIndex=xgBoostModel.OPEN_BOOLEAN_INDEX)
        self.dataTrainRealClean = xgBoostModel.removeZeroInDataTestClean(self.dataTrainClean,
                                                                         openBooleanIndex=xgBoostModel.OPEN_BOOLEAN_INDEX)
        self.labelTrainReal = xgBoostModel.removeZeroInLabel(self.labelTrain)

    def categoricalConversion(self):
        # Categorical Info conversion
        self.dataTrainNumerical, self.dataTestNumerical, self.headerWhole = \
            xgBoostModel.numericalTrainTransformation(self.headerTrainClean, self.dataTrainRealClean,
                                         self.headerTestClean, self.dataTestRealClean)

    def getNumerical(self):
        return self.dataTrainNumerical, self.dataTestNumerical

    def getHeaderWhole(self):
        return self.headerWhole

    def train(self, param, obj=None):
        self.dTrain = xgb.DMatrix(self.dataTrainNumerical, label=self.labelTrainReal)
        self.dTest = xgb.DMatrix(self.dataTestNumerical)
        self.dTrainWithoutLabel = xgb.DMatrix(self.dataTrainNumerical)
        if obj is None:
            bst = xgb.train(params=dict(), dtrain=self.dTrain)
        else:
            bst = xgb.train(params=dict(), dtrain=self.dTrain, obj=obj)
        prediction = bst.predict(self.dTest)
        res = xgBoostModel.restoreZeroEntryInPrediction(prediction, self.dataTestClean,
                                                        openBooleanIndex=xgBoostModel.OPEN_BOOLEAN_INDEX)
        return res, bst



    # data loading and extraction function
    @staticmethod
    def dataLoadExtract(filePath, booleanMin, booleanMax, stateHolidayIndex):
        rawDataMatrix = []
        firstRow = True

        with open(filePath, newline='') as csvFile:
            train_raw = csv.reader(csvFile, delimiter=',')
            for row in train_raw:
                if (firstRow):
                    rawDataMatrix.append(row)
                    firstRow = False
                else:
                    currentRow = []
                    for i in range(len(row)):
                        if (i == stateHolidayIndex):
                            currentRow.append(row[i])
                        elif booleanMin <= i <= booleanMax:
                            if (row[i] == '0'):
                                currentRow.append(False)
                            else:
                                currentRow.append(True)
                        elif '-' in row[i]:
                            currentRow.append(
                                datetime.strptime(row[i], xgBoostModel.DATE_TIME_FORMAT_REAL).date())
                        else:
                            currentRow.append(int(row[i]))
                    rawDataMatrix.append(currentRow)

        headerRaw = rawDataMatrix[0]  # a list containing all the headers as string
        dataRaw = np.array(rawDataMatrix[1:])  # a numpy array with raw data
        return headerRaw, dataRaw

    @staticmethod
    def storeLoadExtract(filePath):
        rawDataMatrix = []
        firstRow = True

        with open(filePath, newline='') as csvFile:
            train_raw = csv.reader(csvFile, delimiter=',')
            for row in train_raw:
                if (firstRow):
                    header = ["Store Index", "Store Type", "Assortment", "Competition distance reciprocal",
                              "Competition Since",
                              "Promotion Since", "Promotion Interval"]
                    rawDataMatrix.append(header)
                    firstRow = False
                else:
                    currentRow = []
                    currentRow.append(int(row[0]))  # store index
                    currentRow.append(row[1])  # store type
                    currentRow.append(row[2])  # assortment

                    if (row[3] == ""):  # competition distance reciprocal
                        currentRow.append(0)
                        currentRow.append(xgBoostModel.STORE_NO_COMPETITION_SINCE_CONSTANT_TIME)
                    else:
                        currentRow.append(1.0 / int(row[3]))

                        if (row[4] != ""):
                            date_str = "1/" + row[4] + "/" + row[5]
                            date_object = datetime.strptime(date_str, xgBoostModel.DATE_TIME_FORMAT_DEV).date()
                            currentRow.append(date_object)  # competition since time
                        else:
                            currentRow.append(xgBoostModel.STORE_COMPETITION_SINCE_DEFAULT_TIME)

                    if (row[6] == "0"):  # promotion specs
                        currentRow.append(xgBoostModel.STORE_NO_PROMOTION_SINCE_CONSTANT_TIME)
                        currentRow.append(xgBoostModel.STORE_NO_PROMO_INTERVAL_STRING)
                    else:
                        date_str = row[8] + "-W" + row[7] + "-0"
                        date_object = datetime.strptime(date_str, xgBoostModel.DATE_TIME_FORMAT_WEEK).date()
                        currentRow.append(date_object)
                        currentRow.append(row[9])

                    rawDataMatrix.append(currentRow)

        headerRaw = rawDataMatrix[0]  # a list containing all the headers as string
        dataRaw = np.array(rawDataMatrix[1:])  # a numpy array with raw data
        return headerRaw, dataRaw

    @staticmethod
    def storeInfoConverter(headerStore, dataStore, dataTrain):
        newHeaderStore = headerStore.copy()  # header processing
        newHeaderStore += ["Average Sales Without Promotion",
                           "Average Sales With Promotion",
                           "Average Sales",
                           "Variance Sales Without Promotion",
                           "Variance Sales With Promotion",
                           "Variance Sales",
                           "Average SC Ratio Without Promotion",
                           "Average SC Ratio With Promotion",
                           "Average SC Ratio",
                           "Variance SC Ratio Without Promotion",
                           "Variance SC Ratio With Promotion",
                           "Variance SC Ratio",
                           "Median Sales Without Promotion",
                           "Median Sales With Promotion",
                           "Median Sales",
                           "Average Open Ratio"]

        resultant_list = []
        for row in dataStore:
            currentRow = list(row.copy())

            # TO DO : maybe we should process the original raw categorical data here

            currentStore = int(row[0])
            sales_list_without_promotion = []
            sales_list_with_promotion = []
            sc_ratio_list_without_promotion = []
            sc_ratio_list_with_promotion = []
            openDayCount = 0
            entryCount = 0

            for salesRow in dataTrain:
                if (int(salesRow[0]) == currentStore):  # It's the store we want to analyze in this round

                    # sales centric
                    if (salesRow[3] > 0):
                        sc_ratio = salesRow[3] / salesRow[4]
                        if (salesRow[6] is True):
                            sales_list_with_promotion.append(salesRow[3])
                            sc_ratio_list_with_promotion.append(sc_ratio)
                        else:
                            sales_list_without_promotion.append(salesRow[3])
                            sc_ratio_list_without_promotion.append(sc_ratio)

                    # open centric
                    if (salesRow[5] is True):
                        openDayCount += 1
                entryCount += 1

            # data processing and adding
            currentRow.append(st.mean(sales_list_without_promotion))
            currentRow.append(st.mean(sales_list_with_promotion))
            currentRow.append(st.mean(sales_list_with_promotion + sales_list_without_promotion))
            currentRow.append(st.variance(sales_list_without_promotion))
            currentRow.append(st.variance(sales_list_with_promotion))
            currentRow.append(st.variance(sales_list_with_promotion + sales_list_without_promotion))
            currentRow.append(st.mean(sc_ratio_list_without_promotion))
            currentRow.append(st.mean(sc_ratio_list_with_promotion))
            currentRow.append(st.mean(sc_ratio_list_with_promotion + sc_ratio_list_without_promotion))
            currentRow.append(st.variance(sc_ratio_list_without_promotion))
            currentRow.append(st.variance(sc_ratio_list_with_promotion))
            currentRow.append(st.variance(sc_ratio_list_with_promotion + sc_ratio_list_without_promotion))
            currentRow.append(st.median(sales_list_without_promotion))
            currentRow.append(st.median(sales_list_with_promotion))
            currentRow.append(st.median(sales_list_with_promotion + sales_list_without_promotion))
            currentRow.append(float(openDayCount / entryCount))

            # Recording of data
            resultant_list.append(currentRow)
        return newHeaderStore, np.array(resultant_list)

    @staticmethod
    def trainDataLabelSplit(dataTrain, headerTrain):
        # This function makes the training data exactly in the same format of testing data read by our functions
        dataTrainNew = np.hstack((dataTrain[:,:3], dataTrain[:,4:])) # remove the sales column
        return dataTrainNew, headerTrain[:3]+headerTrain[4:], dataTrain[:,3] # return data, header and label

    @staticmethod
    def storeInfoPromotionIntervalConverter(string):
        if string == 'Jan,Apr,Jul,Oct':
            return [1, 4, 7, 10]
        elif string == 'Feb,May,Aug,Nov':
            return [2, 5, 8, 11]
        elif string == 'Mar,Jun,Sep,Dec':
            return [3, 6, 9, 12]
        else:
            return []


    @staticmethod
    def storeNaturalJoin(dataStore, data, headerStore, header):
        # In this function, we modify it so that it will join up the promotion `True` store with the info with promotion
        # And the non promotion store with the info with non promotion (info being the average, variance and etc)

        # Header processing
        headerProcessed = ["Day",
                           "Month",
                           "Year",
                           "Month In Promotion"]
        newHeader = [header[1]] + header[3:] + headerProcessed + headerStore[1:4] + \
                    ["Competition Since Day Count", "Promotion Since Day Count",
                     headerStore[6]]  # no repetitive store index in header
        newHeader += [headerStore[9], headerStore[12], headerStore[15],
                      headerStore[18], headerStore[21], headerStore[22]]

        resultant_list = []

        for row in data:
            currentIndex = row[0]
            currentStoreInfo = dataStore[currentIndex - 1,
                               :]  # get corresponding store entry with store index to be removed later
            currentDate = row[2]  # we will get the current datetime object
            currentDay = currentDate.day  # day value (integer)
            currentMonth = currentDate.month  # month value (integer)
            currentYear = currentDate.year  # year value (integer)
            monthInPromotion = currentMonth in xgBoostModel.storeInfoPromotionIntervalConverter(currentStoreInfo[6])  # boolean

            listRow = row.copy().tolist()
            listRow = [listRow[1], ] + listRow[3:]
            currentRow = listRow + [currentDay, currentMonth, currentYear, monthInPromotion]  # new entries

            competitionSinceDate = currentStoreInfo[4]  # competition since date
            promotionSinceDate = currentStoreInfo[5]  # promotion since date
            competitionPastDayCount = (currentDate - competitionSinceDate).days
            competitionPastDayCount = 0 if competitionPastDayCount < 0 else competitionPastDayCount
            promotionPastDayCount = (currentDate - promotionSinceDate).days
            promotionPastDayCount = 0 if promotionPastDayCount < 0 else promotionPastDayCount

            # currentRow += list(currentStoreInfo[1:])
            # concatenate the relevant info (distinguishing promotion and non-promotion)
            promotionBoolean = row[5]
            constantInfo = [currentStoreInfo[1], currentStoreInfo[2], currentStoreInfo[3],
                            competitionPastDayCount, promotionPastDayCount, currentStoreInfo[6]]
            noPromotionList = [currentStoreInfo[7], currentStoreInfo[10], currentStoreInfo[13],
                               currentStoreInfo[16], currentStoreInfo[19], currentStoreInfo[22]]
            promotionList = [currentStoreInfo[8], currentStoreInfo[11], currentStoreInfo[14],
                             currentStoreInfo[17], currentStoreInfo[20], currentStoreInfo[22]]

            # Check and append differently
            if promotionBoolean:
                currentRow += constantInfo + promotionList
            else:
                currentRow += constantInfo + noPromotionList

            # Problem here: we have to append one object to make the numpy array conversion correct
            currentRow.append(object())  # random python object

            resultant_list.append(currentRow)

        # manually remove this object to keep the correctness of the data
        resultNumpyArray = np.array(resultant_list)
        resultNumpyArray = resultNumpyArray[:, 0: resultNumpyArray.shape[1] - 1]

        return newHeader, resultNumpyArray

    @staticmethod
    def removeZeroInDataTestClean(dataTestClean, openBooleanIndex):
        resultantList = []
        for row in dataTestClean:
            if (row[openBooleanIndex] is True):  # the store is opened on that day
                resultantList.append(list(row))
        return np.array(resultantList)


    @staticmethod
    def removeZeroInLabel(label):
        result = []
        for i in label:
            if i != 0:
                result.append(i)
        return np.array(result)

    @staticmethod
    def singleFeatureOneHotKeyEncoder(feature_column_vector_train, feature_column_vector_test):
        # numerical encoding
        enc = LabelEncoder()
        featureListTrain = list(feature_column_vector_train)
        featureListTest = list(feature_column_vector_test)

        # we should fit the one has larger value set
        if (len(set(featureListTrain)) > len(set(featureListTest))):
            enc.fit(featureListTrain)
        else:
            enc.fit(featureListTest)

        labelEncodedFeatureTrain = enc.transform(featureListTrain).reshape(-1, 1)
        labelEncodedFeatureTest = enc.transform(featureListTest).reshape(-1, 1)

        # oneHot encoding
        enc = OneHotEncoder()
        enc.fit(labelEncodedFeatureTrain)  # use train to fit the data

        # return order: train, test
        return enc.transform(labelEncodedFeatureTrain).toarray(), enc.transform(labelEncodedFeatureTest).toarray()

    @staticmethod
    def singleDateTimeColumnNumericalTransformer(dateTimeFeatureColumn):
        processed = np.array([(t - datetime(1970, 1, 1)).total_seconds() / 10 ** 10 for t in dateTimeFeatureColumn])
        return processed.reshape(len(processed), )

    @staticmethod
    def singleDateOnlyColumnNumericalTransformer(dateTimeFeatureColumn):
        processed = np.array([(t - date(1970, 1, 1)).total_seconds() / 10 ** 9 for t in dateTimeFeatureColumn])
        return processed.reshape(len(processed), )

    @staticmethod
    def numericalTrainTransformation(headerTrain, dataTrain, headerTest, dataTest):
        # This function prepares from the clean data to all numerical numpy array ready to be feed into DMatrix
        # We will finally safely remove the store index column

        headerNew = ["DayOfWeek1", "DayOfWeek2", "DayOfWeek3", "DayOfWeek4",
                     "DayOfWeek5", "DayOfWeek6", "DayOfWeek7",
                     "Number of Customers", "Open -dummy -True by default",
                     "PromoBoolean1", "PromoBoolean2", "StateHoliday1",
                     "StateHoliday2", "StateHoliday3",
                     "SchoolHolidayBoolean", "SchoolHolidayBoolean",
                     "Day", "Month", "Year", "Month In Promotion 1", "Month In Promotion 2",
                     "Store type 1", "store type 2", "store type 3", "store type 4",
                     "assortmentType1", "assortmentType2", "assortmentType3",
                     "Competition distance reciprocal", "competitionSinceDayCount",
                     "promotionSinceDayCount", "promotionInterval1", "promotionInterval2",
                     "promotionInterval3", "promotionInterval4",
                     'Average Sales',
                     'Variance Sales',
                     'Average SC Ratio',
                     'Variance SC Ratio',
                     'Median Sales',
                     'Average Open Ratio']

        # This function should get the training features dict for oneHotEncoded features (key -> number of types)


        # Categorical features oneHotKey encoding column


        dayOfWeekColumnTrain, dayOfWeekColumnTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 0], dataTest[:, 0])
        customersNumberColumnTrain, customersNumberColumnTest = dataTrain[:, 1], dataTest[:, 1]

        # the date shouldn't affect the prediction
        # dateDatetimeColumnTrain = singleDateOnlyColumnNumericalTransformer(dataTrain[:,2])
        # dateDatetimeColumnTest = singleDateOnlyColumnNumericalTransformer(dataTest[:,2])

        openBooleanTrain, openBooleanTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 2], dataTest[:, 2])
        promoBooleanTrain, promoBooleanTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 3], dataTest[:, 3])
        stateHolidayBooleanTrain, stateHolidayBooleanTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 4],
                                                                                          dataTest[:, 4])
        schoolHolidayBooleanTrain, schoolHolidayBooleanTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 5],
                                                                                            dataTest[:, 5])

        dayColumnTrain, dayColumnTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 6], dataTest[:, 6])
        monthColumnTrain, monthColumnTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 7], dataTest[:, 7])
        yearColumnTrain, yearColumnTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 8], dataTest[:, 8])

        monthInPromotionBooleanTrain, monthInPromotionBooleanTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 9],
                                                                                                  dataTest[:, 9])
        storeTypeColumnTrain, storeTypeColumnTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 10], dataTest[:, 10])
        assortmentTypeColumnTrain, assortmentTypeColumnTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 11],
                                                                                            dataTest[:, 11])

        competitionDistanceReciprocalTrain, competitionDistanceReciprocalTest = dataTrain[:, 12], dataTest[:, 12]
        competitionSinceDayCountTrain, competitionSinceDayCountTest = dataTrain[:, 13], dataTest[:, 13]
        promotionSinceDayCountTrain, promotionSinceDayCountTest = dataTrain[:, 14], dataTest[:, 14]

        promotionIntervalColumnTrain, promotionIntervalColumnTest = xgBoostModel.singleFeatureOneHotKeyEncoder(dataTrain[:, 15],
                                                                                                  dataTest[:, 15])

        calculatedStatisticsColumnTrain, calculatedStatisticsColumnTest = dataTrain[:, 16:], dataTest[:, 16:]

        # use numpy.column_stack to accomplish column and matrix side by side stacking
        resultantArrayTrain = np.column_stack((dayOfWeekColumnTrain,
                                               customersNumberColumnTrain,
                                               openBooleanTrain,
                                               promoBooleanTrain,
                                               stateHolidayBooleanTrain,
                                               schoolHolidayBooleanTrain,
                                               dayColumnTrain,
                                               #monthColumnTrain,
                                               #yearColumnTrain,
                                               monthInPromotionBooleanTrain,
                                               storeTypeColumnTrain,
                                               assortmentTypeColumnTrain,
                                               competitionDistanceReciprocalTrain,
                                               competitionSinceDayCountTrain,
                                               promotionSinceDayCountTrain,
                                               promotionIntervalColumnTrain,
                                               calculatedStatisticsColumnTrain))

        resultantArrayTest = np.column_stack((dayOfWeekColumnTest,
                                              customersNumberColumnTest,
                                              openBooleanTest,
                                              promoBooleanTest,
                                              stateHolidayBooleanTest,
                                              schoolHolidayBooleanTest,
                                              dayColumnTest,
                                              #monthColumnTest,
                                              #yearColumnTest,
                                              monthInPromotionBooleanTest,
                                              storeTypeColumnTest,
                                              assortmentTypeColumnTest,
                                              competitionDistanceReciprocalTest,
                                              competitionSinceDayCountTest,
                                              promotionSinceDayCountTest,
                                              promotionIntervalColumnTest,
                                              calculatedStatisticsColumnTest))

        return resultantArrayTrain, resultantArrayTest, headerNew  # order: train, test, header

    @staticmethod
    def restoreZeroEntryInPrediction(prediction, dataTestCleanWithZero, openBooleanIndex):
        resultantList = []
        predictionIndex = 0
        for i in range(dataTestCleanWithZero.shape[0]):
            if (dataTestCleanWithZero[i][openBooleanIndex] is True):  # the store is open, put our prediction inside
                resultantList.append(prediction[predictionIndex])
                predictionIndex += 1  # update prediction index to the next prediction point
            else:
                resultantList.append(0)  # the store is closed, append zero and do not update the index
        return np.array(resultantList).reshape(len(resultantList), )




