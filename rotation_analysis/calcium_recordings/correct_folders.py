import sys, os
import shutil

from calcium_recordings.recording import Recording
from margrie_libs.utils.folder_operations import folders_starting_with


exampleOriginalPath = "\(20160108_18_57_12\)__spont_100x100ym_256x256pxl/\(20160108_18_57_12\)__spont_100x100ym_256x256pxl.raw"


def extract_folder_base_name(folderBaseName):
    """
    Gives a pure folder base name for depth (without / and \\)
    :param folderBaseName:
    :return:
    """
    if folderBaseName[-1] in ('/', '\\'):
        folderBaseName = folderBaseName[:-1]
    if folderBaseName[0] in ('/', '\\'):
        folderBaseName = folderBaseName[1:]
    if folderBaseName[-2:] != 'um':
        print('{} is not a depth folder (does not end with "um")'.format(folderBaseName))
        return False
    else:
        folderBaseName = folderBaseName[:-2]
    return folderBaseName


def isDepthFolder(folderBaseName):
    folderBaseName = extract_folder_base_name(folderBaseName)
    if not folderBaseName:
        return False
    try:
        float(folderBaseName)
        return True
    except ValueError:
        return False


def getDepthDirs(srcDir):
    base_list = os.listdir(srcDir)  # To ensure both start from same list as listdir not predictable
    values_list = [os.path.join(srcDir, d) for d in base_list if isDepthFolder(d)]
    sort_keys_list = [extract_folder_base_name(baseName) for baseName in base_list]
    return [val for (key, val) in sorted(zip(sort_keys_list, values_list))]


def removeUselessFolder(srcDir):
    depthDirs = getDepthDirs(srcDir)
    for d in depthDirs:
        uselessDir = os.path.join(d, os.listdir(d)[0])
        
        usefullDirs = os.listdir(os.path.join(d, uselessDir))
        for usefullDir in usefullDirs:
            realUsefulPath = os.path.join(uselessDir, usefullDir)
            newUsefullPath = os.path.join(d, usefullDir)
            if __debug__:
                print("Would move {} to {}".format(realUsefulPath, newUsefullPath))
            else:
                shutil.move(realUsefulPath, newUsefullPath)
        os.rmdir(uselessDir)


def getDataFolders(srcDir):
    """
    Meant to be used before renaming (find strategy won't work afterwards)
    """
    dataFoldersPaths = []
    depthDirs = getDepthDirs(srcDir)
    for d in depthDirs:
        tmpDataFolders = folders_starting_with(d, '(')
        dataFoldersPaths.append(tmpDataFolders)
    return dataFoldersPaths    

def getMaxAnglesTypes(recordings):
    return set([r.getMaxAngle() for r in recordings])
    
def setRecNb(rec, nb):
    rec.setRecNb(nb)
    return rec
    
def setRecNbs(recordings):
    maxAngleTypes = getMaxAnglesTypes(recordings)
    sortedRecsDict = {}
    for angle in maxAngleTypes:
        crntAngleRecs = [r for r in recordings if r.getMaxAngle() == angle]
        crntAngleRecs.sort(key=lambda x: x.dateTime) # sort by acquisition time

        crntHighRes = [r for r in crntAngleRecs if r.isHighRes()]
        crntRecs = [r for r in crntAngleRecs if r.isRec()]
        crntStacks = [r for r in crntAngleRecs if r.isZStack()]
        crntSortedRecsDict = {
            'highResRecs': [setRecNb(rec, i) for i, rec in enumerate(crntHighRes)],
            'recs': [setRecNb(rec, i) for i, rec in enumerate(crntRecs)],
            'stacks': [setRecNb(rec, i) for i, rec in enumerate(crntStacks)]
            }
        sortedRecsDict[angle] = crntSortedRecsDict
    return sortedRecsDict
    
def makeAnglesDirs(depthDir, recordings):
    maxAngles = getMaxAnglesTypes(recordings)
    for angle in maxAngles:
        os.mkdir(os.path.join(depthDir, "maxAngle{}".format(int(angle))))    

def main(srcDir):
    """
    MAIN FUNCTION
    """
    srcDir = os.path.abspath(srcDir)
    removeUselessFolder(srcDir)
    depthDirs = getDepthDirs(srcDir)
    dataFolders = getDataFolders(srcDir)

    for i, depthDir in enumerate(depthDirs):
        recs = []
        for folder in dataFolders[i]:
            try:
                recs.append(Recording(folder))
            except RuntimeError: # Could not create instance
                print("Could not create recording for folder {}, skipping".format(folder))
        setRecNbs(recs)

        makeAnglesDirs(depthDir, recs)
        [rec.move() for rec in recs]

        for rec in recs:
            rec.writeIni()

if __name__ == "__main__":
    main(sys.argv[1])
