#!/bin/python
import pandas as pd
import numpy as np
import csv
import math

pd.set_option('display.line_width', 300)
df = pd.read_csv('all_season_data.csv')

firstPlayOfDriveIndex = df.ix[:,[0,4]]
data = [{'gameid' : 0, 'off': 0}]
rowVec = pd.DataFrame(data)
firstPlayOfDriveIndexTwo = firstPlayOfDriveIndex.append(rowVec, ignore_index=True)
firstPlayOfDriveIndexThree = rowVec.append(firstPlayOfDriveIndex, ignore_index=True)
firstPlayOfDriveIndexThree.columns = ['secGameID','secOff']
firstPlayOfDriveIndexTwo = pd.DataFrame(firstPlayOfDriveIndexTwo)
firstPlayOfDriveIndexThree = pd.DataFrame(firstPlayOfDriveIndexThree)
firstPlayMaster = firstPlayOfDriveIndexTwo.join(firstPlayOfDriveIndexThree)

#print firstPlayMaster.head(6)
firstPlayMaster = pd.DataFrame(firstPlayMaster)
#first = firstPlayMaster.drop(firstPlayMaster['off']==firstPlayMaster['secOff'])
#firstPlayMaster['firstDown'] = firstPlayMaster['off']!=firstPlayMaster['secOff'] 
firstPlayMaster['offTF'] = firstPlayMaster['off'] == firstPlayMaster['secOff'] 
firstPlayMaster['gameIDTF'] = firstPlayMaster['gameid']==firstPlayMaster['secGameID']
firstPlay = firstPlayMaster[firstPlayMaster['offTF'] == False]
firstPlayTwo = firstPlay[firstPlayMaster['gameIDTF'] == True]
indexValsFirst = firstPlayTwo.index
vec = [1]*len(indexValsFirst)

lastIndexData = pd.DataFrame({'index' : indexValsFirst, 'vec' : vec})
indexValsLastOne = lastIndexData['index'] - lastIndexData['vec']
lastEle = len(indexValsLastOne)
lastElement = len(df['gameid'])-1
indexValsLast = indexValsLastOne[1:lastEle]
#indexValsLast = np.array(indexValsLast)
indexValsLastTwo = np.append(indexValsLast, lastElement)
firstPlay = df.ix[indexValsFirst, :]
lastPlay = df.ix[indexValsLastTwo, :]
firstPlay['mergeVar'] = np.arange(len(firstPlay))
lastPlay['mergeVar'] = np.arange(len(firstPlay))

driveByDrive = firstPlay.merge(lastPlay, on = 'mergeVar', suffixes = ('_first', '_last')) 
print driveByDrive
#driveByDrive.to_csv('driveByDrive.csv', sep=',')
