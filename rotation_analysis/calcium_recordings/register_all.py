import os
import pprint
from argparse import ArgumentParser

from subprocess import check_call
from collections import OrderedDict as ODict

from margrie_libs.utils.folder_operations import folders_starting_with

from calcium_recordings.correct_folders import getDepthDirs, setRecNbs
from calcium_recordings.recording import Recording


def getParentFolder(folder):
    return os.path.dirname(os.path.normpath(folder))


def getFolderBaseName(folder):
    basename = os.path.basename(os.path.normpath(folder))
    basename = basename.strip()
    return basename


def getDepth(folder):
    basename = getFolderBaseName(folder)
    return int(basename.replace('um', ''))


def getDataFolders(srcDir):
    dataFoldersPaths = []
    depths = []
    for d in getDepthDirs(srcDir):
        depth = getDepth(d)
        angleFolders = folders_starting_with(d, 'maxAngle')
        print("Number of angles at {}um: {}".format(depth, len(angleFolders)))
        for angleFolder in angleFolders:
            recFolders = folders_starting_with(angleFolder, 'rec')
            depths.append(depth)  # So that it matches in items with dataFoldersPaths
            dataFoldersPaths.append(recFolders)
    return depths, dataFoldersPaths


def getRecsTree(srcDir):
    depths, dataFolders = getDataFolders(srcDir)
    recsTree = ODict()
    for depth, dataFoldersAtDepth in zip(depths, dataFolders):
        recs = []
        if not depth in recsTree.keys():
            recsTree[depth] = {}
        for folder in dataFoldersAtDepth:
            try:
                recs.append(Recording(folder))
            except RuntimeError as err:  # Could not create instance
                print("Warning: could not create recording for folder {}, skipping; {}".format(folder, err))
        numberedRecs = setRecNbs(recs)
        if numberedRecs:  # Skip if nothing of type 'rec'
            try:
                angle = list(numberedRecs.keys())[0]
                numberedRecsDict = list(numberedRecs.values())[0]
            except IndexError:
                print("Depth: {}".format(depth))
                print("\tRecordings: {}".format(numberedRecs))
                raise
            recsTree[depth][angle] = numberedRecsDict
    return recsTree


def main(srcDir, channelsToProcess, refChannel, overwrite, gaussianKernelSize, baselineN):
    """
    MAIN FUNCTION
    """
    
    fijiPrefix = 'fiji --allow-multiple '
    macroPath = 'register.ijm'
    fijiCmd = fijiPrefix + ' -batch ' + macroPath
    
    srcDir = os.path.abspath(srcDir)
    recsTree = getRecsTree(srcDir)
    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(recsTree)
    
    for depth, recsAtDepth in recsTree.items():
        print("Processing depth: {}".format(depth))
        if not recsAtDepth:
            continue
        for angle in recsAtDepth.keys():
            print("\tProcessingangle: {}".format(angle))
            for rec in recsAtDepth[angle]['recs']: # Skip the highRes and stacks
                nChannels = len(rec.derotatedImgFilesPaths)
                if  len(channelsToProcess) <= nChannels:
                    
                    process = False
                    if overwrite:
                        process = True
                    else:
                        if not(rec.registeredImgFilesPaths):
                            process = True
                    
                    if process:
                        if baselineN == 'whole':
                            nRefImgs = rec.getNFrames()
                        elif baselineN == 'bsl':
                            nRefImgs = rec.getNBslFrames()
                        else:
                            raise ValueError("Expected one of ['whole', 'bsl'], got {}.".format(baselineN))
                        print("Registering images in {}".format(rec.dir))
                        imgPath1 = rec.derotatedImgFilesPaths[channelsToProcess[0]]
                        if len(channelsToProcess) > 1:
                            imgPath2 = rec.derotatedImgFilesPaths[channelsToProcess[1]]
                        else:
                            imgPath2 = imgPath1
                        wholeCmd = '{} {},{},{},{},{}'.format(fijiCmd,
                                                        rec.derotatedImgFilesPaths[refChannel],
                                                        imgPath1,
                                                        imgPath2,
                                                        nRefImgs,
                                                        gaussianKernelSize)
                        check_call(wholeCmd, shell=True)
                    else:
                        print("Info: Recording {} already processed, skipping as instructed".format(rec))
                else:
                    print("Warning: Skipping recording {}, channel missing, got {} ou of {}".format(rec, nChannels, len(channelsToProcess)))


if __name__ == "__main__":
    program_name = os.path.basename(__file__)
    parser = ArgumentParser(prog=program_name, description='Program to recursively register (Fiji turboreg) all recordings in an experiment')
    parser.add_argument("-b", "--baseline-type", dest="baselineN", type=str, choices=('whole', 'bsl'), default='whole', help="The type of baseline to use (one of %(choices)s). Default: %(default)s")
    parser.add_argument("-k", "--kernel-size", dest="gaussianKernelSize", type=float, default=1.2,  help="The size of the gaussian kernel to use for filtering prior to registration. Default: %(default)s")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Do we overwrite recordings that have already been processed")
    parser.add_argument("-r", "--reference-channel", dest="refChannel", type=int, choices=(1, 2), default=1, help="The reference channel. Default: %(default)s")
    parser.add_argument("-p", "--channels-to-process", dest="channelsToProcess", type=int, nargs='+', default=[1, 2], help="The list of channels to process. Default: %(default)s")
    parser.add_argument('source_directory', type=str, help='The source directory of the experiment')

    args = parser.parse_args()

    if len(args.channelsToProcess) > 2:
        raise ValueError("Channels to process has to be a list of 1 or 2 elements, got {}: {}".format(len(args.channelsToProcess), args.channelsToProcess))
    
    ## Translate to list indexing
    refChannel = args.refChannel -1
    channelsToProcess = [c-1 for c in args.channelsToProcess]
    
    main(args.source_directory, channelsToProcess, refChannel, args.overwrite, args.gaussianKernelSize, args.baselineN)
