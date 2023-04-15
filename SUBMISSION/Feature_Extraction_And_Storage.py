import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

empty = []
emptyset = np.array(empty)

#csv_file = "Feature_Extraction_Data.csv"
#info = ["avgXAccel", "maxXAccel",  "avgYAccel", "maxYAccel", "avgZAccel" ,"maxZAccel", "avgTimeAboveAvgZ", "avgAccel" , "maxAccel", "avgTimeAboveAvg", "label"]

#df.to_csv(csv_file, mode='a', header=False, index=False)


df = pd.DataFrame([emptyset])

with h5.File('./data.h5', 'r') as hdf:
    group = hdf.get('Ethan/Jump')
    group = list(group.keys())
for i in range (1, len(group)+1):
        with h5.File('./data.h5', 'r') as hdf:
            dataset = hdf.get('Ethan/Jump/' + str(group[i - 1]))
            myarray = np.array(dataset)
        #print(dataset)
        for i in range(0, len(myarray)):
            if i == 0:
                myarray[i][1] = myarray[i][1]
                myarray[i][2] = myarray[i][2]
                myarray[i][3] = myarray[i][3]
            else:
                myarray[i][1] = (myarray[i][1] + myarray[i - 1][1] + myarray[i - 2][1] + myarray[i - 3][1]) / 4
                myarray[i][2] = (myarray[i][2] + myarray[i - 1][2] + myarray[i - 2][2] + myarray[i - 3][2]) / 4
                myarray[i][3] = (myarray[i][3] + myarray[i - 1][3] + myarray[i - 2][3] + myarray[i - 3][3]) / 4

        # Average total acceleration
        avgAccel = 0
        for i in range(0, len(myarray)):
            avgAccel += abs(myarray[i][4] / 500)

        # max total acceleration
        maxAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][4]) > maxAccel:
                maxAccel = abs(myarray[i][4])

        # Average X acceleration
        avgXAccel = 0
        for i in range(0, len(myarray)):
            avgXAccel += myarray[i][1] / 500

        # Average Y acceleration
        avgYAccel = 0
        for i in range(0, len(myarray)):
            avgYAccel += myarray[i][2] / 500

        # Average Z acceleration
        avgZAccel = 0
        for i in range(0, len(myarray)):
            avgZAccel += abs(myarray[i][3] / 500)
        # average time above avg Z acceleration

        avgZTime = 0
        timer = False
        times = 0
        totalTime = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][3]) >= avgZAccel:
                if timer == False:
                    timer = True
                    times += 1
                    totalTime += 1
                else:
                    totalTime += 1
            else:
                timer = False

        avgZTime = totalTime / times

        # max Z acceleration
        maxZAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][3]) > maxZAccel:
                maxZAccel = abs(myarray[i][3])

        # max X acceleration
        maxXAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][1]) > maxXAccel:
                maxXAccel = abs(myarray[i][1])

        # max Y acceleration
        maxYAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][2]) > maxYAccel:
                maxYAccel = abs(myarray[i][2])

        # average time above avg acceleration

        avgTime = 0
        timer = False
        times = 0
        totalTime = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][4]) >= abs(avgAccel):
                if timer == False:
                    timer = True
                    times += 1
                    totalTime += 1
                else:
                    totalTime += 1
            else:
                timer = False

        avgTime = totalTime / times


        info = [avgXAccel, maxXAccel, avgYAccel, maxYAccel, avgZAccel, maxZAccel, avgZTime, avgAccel, maxAccel, avgTime, 1]
        df = pd.concat([df, pd.DataFrame([info])])
        #df.to_csv(csv_file, mode='a', header=False, index=False)



with h5.File('./data.h5', 'r') as hdf:
    group = hdf.get('Ethan/Walk')
    group = list(group.keys())
for i in range (1, len(group)+1):
        with h5.File('./data.h5', 'r') as hdf:
            dataset = hdf.get('Ethan/Walk/' + str(group[i - 1]))
            myarray = np.array(dataset)
        #print(dataset)
            for i in range(0, len(myarray)):
                if i == 0:
                    myarray[i][1] = myarray[i][1]
                    myarray[i][2] = myarray[i][2]
                    myarray[i][3] = myarray[i][3]
                else:
                    myarray[i][1] = (myarray[i][1] + myarray[i - 1][1] + myarray[i - 2][1] + myarray[i - 3][1]) / 4
                    myarray[i][2] = (myarray[i][2] + myarray[i - 1][2] + myarray[i - 2][2] + myarray[i - 3][2]) / 4
                    myarray[i][3] = (myarray[i][3] + myarray[i - 1][3] + myarray[i - 2][3] + myarray[i - 3][3]) / 4

            # Average total acceleration
            avgAccel = 0
            for i in range(0, len(myarray)):
                avgAccel += myarray[i][4] / 500
            # max total acceleration
            maxAccel = 0
            for i in range(0, len(myarray)):
                if abs(myarray[i][4]) > maxAccel:
                    maxAccel = abs(myarray[i][4])

            # Average X acceleration
            avgXAccel = 0
            for i in range(0, len(myarray)):
                avgXAccel += myarray[i][1] / 500

            # Average Y acceleration
            avgYAccel = 0
            for i in range(0, len(myarray)):
                avgYAccel += myarray[i][2] / 500

            # Average Z acceleration
            avgZAccel = 0
            for i in range(0, len(myarray)):
                avgZAccel += abs(myarray[i][3] / 500)
            # average time above half max acceleration

            # max Z acceleration
            maxZAccel = 0
            for i in range(0, len(myarray)):
                if abs(myarray[i][3]) > maxZAccel:
                    maxZAccel = abs(myarray[i][3])

            # max X acceleration
            maxXAccel = 0
            for i in range(0, len(myarray)):
                if abs(myarray[i][1]) > maxXAccel:
                    maxXAccel = abs(myarray[i][1])

            # max Y acceleration
            maxYAccel = 0
            for i in range(0, len(myarray)):
                if abs(myarray[i][2]) > maxYAccel:
                    maxYAccel = abs(myarray[i][2])

            # average time above avg acceleration

            avgTime = 0
            timer = False
            times = 0
            totalTime = 0
            for i in range(0, len(myarray)):
                if abs(myarray[i][4]) >= abs(avgAccel):
                    if timer == False:
                        timer = True
                        times += 1
                        totalTime += 1
                    else:
                        totalTime += 1
                else:
                    timer = False

            avgTime = totalTime / times

        info = [avgXAccel, maxXAccel, avgYAccel, maxYAccel, avgZAccel, maxZAccel, avgZTime, avgAccel, maxAccel, avgTime, 0]
        df = pd.concat([df, pd.DataFrame([info])])



with h5.File('./data.h5', 'r') as hdf:
    group = hdf.get('Jacob/Jump')
    group = list(group.keys())
for i in range(1, len(group) + 1):
    with h5.File('./data.h5', 'r') as hdf:
        dataset = hdf.get('Jacob/Jump/' + str(group[i - 1]))
        myarray = np.array(dataset)
    #print(dataset)
    for i in range (0, len(myarray)):
        if i == 0:
            myarray[i][1] = myarray[i][1]
            myarray[i][2] = myarray[i][2]
            myarray[i][3] = myarray[i][3]
        else:
            myarray[i][1] = (myarray[i][1] + myarray[i-1][1] + myarray[i-2][1] + myarray[i-3][1])/4
            myarray[i][2] = (myarray[i][2] + myarray[i-1][2] + myarray[i-2][2] + myarray[i-3][2])/4
            myarray[i][3] = (myarray[i][3] + myarray[i-1][3] + myarray[i-2][3] + myarray[i-3][3])/4

    # Average total acceleration
    avgAccel = 0
    for i in range (0, len(myarray)):
        avgAccel += myarray[i][4]/500
    # max total acceleration
    maxAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][4]) > maxAccel:
            maxAccel = abs(myarray[i][4])

    # Average X acceleration
    avgXAccel = 0
    for i in range (0, len(myarray)):
        avgXAccel += myarray[i][1]/500

    # Average Y acceleration
    avgYAccel = 0
    for i in range (0, len(myarray)):
        avgYAccel += myarray[i][2]/500

    # Average Z acceleration
    avgZAccel = 0
    for i in range (0, len(myarray)):
        avgZAccel += abs(myarray[i][3]/500)
    # max Z acceleration
    maxZAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][3]) > maxZAccel:
            maxZAccel = abs(myarray[i][3])

    # max X acceleration
    maxXAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][1]) > maxXAccel:
            maxXAccel = abs(myarray[i][1])

    # max Y acceleration
    maxYAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][2]) > maxYAccel:
            maxYAccel = abs(myarray[i][2])

    # average time above avg acceleration

    avgTime = 0
    timer = False
    times = 0
    totalTime = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][4]) >= abs(avgAccel):
            if timer == False:
                timer = True
                times += 1
                totalTime += 1
            else:
                totalTime += 1
        else:
            timer = False

    avgTime = totalTime / times

    info = [avgXAccel, maxXAccel, avgYAccel, maxYAccel, avgZAccel, maxZAccel, avgZTime, avgAccel, maxAccel, avgTime, 1]
    df = pd.concat([df, pd.DataFrame([info])])
    #df.to_csv(csv_file, mode='a', header=False, index=False)

with h5.File('./data.h5', 'r') as hdf:
    group = hdf.get('Jacob/Walk')
    group = list(group.keys())
for i in range(1, len(group) + 1):
    with h5.File('./data.h5', 'r') as hdf:
        dataset = hdf.get('Jacob/Walk/' + str(group[i - 1]))
        myarray = np.array(dataset)
    #print(dataset)
        for i in range(0, len(myarray)):
            if i == 0:
                myarray[i][1] = myarray[i][1]
                myarray[i][2] = myarray[i][2]
                myarray[i][3] = myarray[i][3]
            else:
                myarray[i][1] = (myarray[i][1] + myarray[i - 1][1] + myarray[i - 2][1] + myarray[i - 3][1]) / 4
                myarray[i][2] = (myarray[i][2] + myarray[i - 1][2] + myarray[i - 2][2] + myarray[i - 3][2]) / 4
                myarray[i][3] = (myarray[i][3] + myarray[i - 1][3] + myarray[i - 2][3] + myarray[i - 3][3]) / 4

        # Average total acceleration
        avgAccel = 0
        for i in range(0, len(myarray)):
            avgAccel += myarray[i][4] / 500
        # max total acceleration
        maxAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][4]) > maxAccel:
                maxAccel = abs(myarray[i][4])

        # Average X acceleration
        avgXAccel = 0
        for i in range(0, len(myarray)):
            avgXAccel += myarray[i][1] / 500

        # Average Y acceleration
        avgYAccel = 0
        for i in range(0, len(myarray)):
            avgYAccel += myarray[i][2] / 500

        # Average Z acceleration
        avgZAccel = 0
        for i in range(0, len(myarray)):
            avgZAccel += abs(myarray[i][3] / 500)
        # max Z acceleration
        maxZAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][3]) > maxZAccel:
                maxZAccel = abs(myarray[i][3])

        # max X acceleration
        maxXAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][1]) > maxXAccel:
                maxXAccel = abs(myarray[i][1])

        # max Y acceleration
        maxYAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][2]) > maxYAccel:
                maxYAccel = abs(myarray[i][2])

        # average time above avg acceleration

        avgTime = 0
        timer = False
        times = 0
        totalTime = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][4]) >= abs(avgAccel):
                if timer == False:
                    timer = True
                    times += 1
                    totalTime += 1
                else:
                    totalTime += 1
            else:
                timer = False

        avgTime = totalTime / times


        info = [avgXAccel, maxXAccel, avgYAccel, maxYAccel, avgZAccel, maxZAccel, avgZTime, avgAccel, maxAccel, avgTime, 0]
        df = pd.concat([df, pd.DataFrame([info])])
        #df.to_csv(csv_file, mode='a', header=False, index=False)


with h5.File('./data.h5', 'r') as hdf:
    group = hdf.get('Lauren/Jump')
    group = list(group.keys())
for i in range(1, len(group) + 1):
    with h5.File('./data.h5', 'r') as hdf:
        dataset = hdf.get('Lauren/Jump/' + str(group[i - 1]))
        myarray = np.array(dataset)
    # print(dataset)
    for i in range (0, len(myarray)):
        if i == 0:
            myarray[i][1] = myarray[i][1]
            myarray[i][2] = myarray[i][2]
            myarray[i][3] = myarray[i][3]
        else:
            myarray[i][1] = (myarray[i][1] + myarray[i-1][1] + myarray[i-2][1] + myarray[i-3][1])/4
            myarray[i][2] = (myarray[i][2] + myarray[i-1][2] + myarray[i-2][2] + myarray[i-3][2])/4
            myarray[i][3] = (myarray[i][3] + myarray[i-1][3] + myarray[i-2][3] + myarray[i-3][3])/4

    # Average total acceleration
    avgAccel = 0
    for i in range (0, len(myarray)):
        avgAccel += myarray[i][4]/500

    # max total acceleration
    maxAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][4]) > maxAccel:
            maxAccel = abs(myarray[i][4])

    # Average X acceleration
    avgXAccel = 0
    for i in range (0, len(myarray)):
        avgXAccel += myarray[i][1]/500

    # Average Y acceleration
    avgYAccel = 0
    for i in range (0, len(myarray)):
        avgYAccel += myarray[i][2]/500

    # Average Z acceleration
    avgZAccel = 0
    for i in range (0, len(myarray)):
        avgZAccel += abs(myarray[i][3]/500)
    # max Z acceleration
    maxZAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][3]) > maxZAccel:
            maxZAccel = abs(myarray[i][3])

    # max X acceleration
    maxXAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][1]) > maxXAccel:
            maxXAccel = abs(myarray[i][1])

    # max Y acceleration
    maxYAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][2]) > maxYAccel:
            maxYAccel = abs(myarray[i][2])

    # average time above avg acceleration

    avgTime = 0
    timer = False
    times = 0
    totalTime = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][4]) >= abs(avgAccel):
            if timer == False:
                timer = True
                times += 1
                totalTime += 1
            else:
                totalTime += 1
        else:
            timer = False

    avgTime = totalTime / times
    info = [avgXAccel, maxXAccel, avgYAccel, maxYAccel, avgZAccel, maxZAccel, avgZTime, avgAccel, maxAccel, avgTime, 1]
    df = pd.concat([df, pd.DataFrame([info])])
    #df.to_csv(csv_file, mode='a', header=False, index=False)

with h5.File('./data.h5', 'r') as hdf:
    group = hdf.get('Lauren/Walk')
    group = list(group.keys())
for i in range(1, len(group) + 1):
    with h5.File('./data.h5', 'r') as hdf:
        dataset = hdf.get('Lauren/Walk/' + str(group[i - 1]))
        myarray = np.array(dataset)
    #print(dataset)
        for i in range(0, len(myarray)):
            if i == 0:
                myarray[i][1] = myarray[i][1]
                myarray[i][2] = myarray[i][2]
                myarray[i][3] = myarray[i][3]
            else:
                myarray[i][1] = (myarray[i][1] + myarray[i - 1][1] + myarray[i - 2][1] + myarray[i - 3][1]) / 4
                myarray[i][2] = (myarray[i][2] + myarray[i - 1][2] + myarray[i - 2][2] + myarray[i - 3][2]) / 4
                myarray[i][3] = (myarray[i][3] + myarray[i - 1][3] + myarray[i - 2][3] + myarray[i - 3][3]) / 4

        # Average total acceleration
        avgAccel = 0
        for i in range(0, len(myarray)):
            avgAccel += myarray[i][4] / 500

        # max total acceleration
        maxAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][4]) > maxAccel:
                maxAccel = abs(myarray[i][4])

        # Average X acceleration
        avgXAccel = 0
        for i in range(0, len(myarray)):
            avgXAccel += myarray[i][1] / 500

        # Average Y acceleration
        avgYAccel = 0
        for i in range(0, len(myarray)):
            avgYAccel += myarray[i][2] / 500

        # Average Z acceleration
        avgZAccel = 0
        for i in range(0, len(myarray)):
            avgZAccel += abs(myarray[i][3] / 500)
        # max Z acceleration
        maxZAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][3]) > maxZAccel:
                maxZAccel = abs(myarray[i][3])

        # max X acceleration
        maxXAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][1]) > maxXAccel:
                maxXAccel = abs(myarray[i][1])

        # max Y acceleration
        maxYAccel = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][2]) > maxYAccel:
                maxYAccel = abs(myarray[i][2])

        # average time above avg acceleration

        avgTime = 0
        timer = False
        times = 0
        totalTime = 0
        for i in range(0, len(myarray)):
            if abs(myarray[i][4]) >= abs(avgAccel):
                if timer == False:
                    timer = True
                    times += 1
                    totalTime += 1
                else:
                    totalTime += 1
            else:
                timer = False

        avgTime = totalTime / times


        info = [avgXAccel, maxXAccel, avgYAccel, maxYAccel, avgZAccel, maxZAccel, avgZTime, avgAccel, maxAccel, avgTime, 0]
        df = pd.concat([df, pd.DataFrame([info])])
        #df.to_csv(csv_file, mode='a', header=False, index=False)

df.columns = ["avgXAccel", "maxXAccel",  "avgYAccel", "maxYAccel", "avgZAccel" ,"maxZAccel", "avgTimeAboveAvgZ", "avgAccel" , "maxAccel", "avgTimeAboveAvg", "label"]


Train, Test = train_test_split(df, test_size=0.1, shuffle=True)

with h5.File('./project_files.h5', 'w') as hdf:
    group = hdf.create_group('/dataset')
    group.create_dataset('train', data = Train)
    group.create_dataset('test', data = Test)