"""
 Trevor Ridgley
 Dr. Steven Reeves
 AM-148
 Sprint 2020

 Final Project: Parallelizing a genome sequence aligner using Numba

 Test Usage: python simpleMapJitClass.py hg19.chr3.9mb.fa NA12878.ihs.chr3.100kb.1.fastq.tiny --t 30 --log=DEBUG

 Code scaffold by Dr. Benedict Paten, Biomolecular Engineering & Bioinformatics, UC Santa Cruz
"""
import array
import sys
import numpy
import pysam
import argparse
import logging 
import time
logger = logging.getLogger()
from numba import njit, types, typed, typeof
from numba.experimental import jitclass # http://numba.pydata.org/numba-doc/latest/reference/deprecation.html#change-of-jitclass-location

# The MinimizerIndexer jitclass requires explicit signatures for the container attributes
# CITATION: https://stackoverflow.com/questions/56463147/how-to-specify-the-string-data-type-when-using-numba
mm_kv_types = (types.unicode_type, types.ListType(types.int64))
mo_kv_types = (types.unicode_type, types.int64)

# The clas spec tells numba how to pre-compile the instance attributes
miSpec = [
    ('targetString', types.unicode_type),
    ('w', types.int64),
    ('k', types.int64),
    ('t', types.int64),
    ('minimizerMap', types.DictType(*mm_kv_types)),
    ('minmerOccurrences', types.DictType(*mo_kv_types))
]

# Decorates a class for just-in-time pre-compiling by Numba
@jitclass(miSpec)
class MinimizerIndexer(object):
    """ Simple minimizer based substring-indexer. 
    
    Please read: https://doi.org/10.1093/bioinformatics/bth408
    
    Related to idea of min-hash index and other "sketch" methods.
    """
    def __init__(self, targetString, w, k, t):
        """ The target string is a string/array of form "[ACGT]*".
        
        Stores the lexicographically smallest k-mer in each window of length w, such that 
        w >= k positions. This smallest k-mer is termed a minmer. 
        
        If a minmer occurs in the target sequence more than t times as a minmer then it is
        omitted from the index, i.e. if the given minmer (kmer) is a minmer
        in more than t different locations in the target string. Note, a minmer may be the 
        minmer for more than t distinct windows and not be pruned, we remove minmers 
        only if they have more than t distinct occurrences as minmers in the sequence.
        """
        
        self.targetString = targetString
        self.w = w
        self.k = k
        self.t = t # If a minmer occurs more than t times then its entry is removed from the index
        # This is a heuristic to remove repetitive minmers that would create many spurious alignments between
        # repeats
        
        # Hash of minmers to query locations, stored as a map whose keys
        # are minmers and whose values are lists of the start indexes of
        # occurrences of the corresponding minmer in the targetString, 
        # sorted in ascending order of index in the targetString.
        #
        # For example if k = 2 and w = 4 and targetString = "GATTACATTT"
        #
        # GATTACATTT
        # GATT (AT)
        #  ATTA (AT)
        #   TTAC (AC)
        #    TACA (AC)
        #     ACAT (AC)
        #      CATT (AT)
        #       ATTT (AT)
        #
        # then self.minimizerMap = { "AT":(1,6), "AC":(4,) }

        # Declare the container attributes using Numba's pre-defined types
        self.minimizerMap = typed.Dict.empty(*mm_kv_types)
        self.minmerOccurrences = typed.Dict.empty(*mo_kv_types)
        
        # Code to complete to build index - you are free to define additional functions 
        for i in range(len(targetString)-self.w+1): # , self.w):
            # Iterate each possible window of width w in the targetString
            minmer = ""
            candidateMinmers = typed.List() # list()
            for j in range(self.w-self.k+1):
                # Then iterate each possible kmer of length k in the window w
                candidateMinmer = targetString[i+j:i+j+self.k]
                candidateMinmers.append((candidateMinmer, i+j))
            try:
                # Take the lexico min, if present
                minmerTuple = min(candidateMinmers)
                minmer = minmerTuple[0]
                site = minmerTuple[1]
                if 'N' not in minmer:
                    try:
                        # Log the minmer occurrences to consider only rare ones
                        self.minmerOccurrences[minmer] += 1
                    except:
                        # REF: https://numba.pydata.org/numba-doc/latest/user/troubleshoot.html
                        # self.minmerOccurrences[minmer] = [np.float64(x) for x in range(0)] # list()
                        # self.minmerOccurrences[minmer].append(site)
                        self.minmerOccurrences[minmer] = 1
                    if self.minmerOccurrences[minmer] > self.t:
                        # Exclude minmers that are present in more than t sites
                        try:
                            del self.minimizerMap[minmer]
                        except:
                            # CITATION: https://stackoverflow.com/questions/19522990/python-catch-exception-and-continue-try-block
                            pass
                else:
                    #continue
                    # print('Excluded minmer:', minmer)
                    pass
            except:
                # Nothing to add if the candidate list had no min
                # continue
                pass
            try:
                if 'N' not in minmer and site not in self.minimizerMap[minmer]:
                    # Prevent duplicate sites from adjacent windows
                    self.minimizerMap[minmer].append(site)
            except:
                if 'N' not in minmer:
                    self.minimizerMap[minmer] = typed.List([site])

    def getMatches(self, searchString):
        """ Iterates through search string finding minmers in searchString and
        yields their list of minmer occurrences in targetString, each as a pair of (x, (y,)*N),
        where x is the index in searchString and y is an occurrence in targetString.
        
        For example if k = 2 and w = 4 and targetString = "GATTACATTT" and searchString = "GATTTAC"
        then self.minimizerMap = { "AT":(1,6), "AC":(4,) }
        and getMatches will yield the following sequence:
        (1, (1,6)), (5, (4,))
        
        You will need to use the "yield" keyword
        """
        # Code to complete - you are free to define additional functions 
        # Track minmers already returned by the generator to avoid re-processing
        yieldedSites = typed.List.empty_list(types.int64) # list()
        # print("getMatches()", searchString)
        # print("getMatches() with string {0}".format(searchString))
        for i in range(len(searchString)-self.w+1):
            # Iterate each possible window of width w in the targetString
            minmer = "" 
            #  Store each read minmer to be iterated via the generator
            sites = typed.List.empty_list(types.int64)
            for j in range(self.w-self.k+1):
                # Then iterate each possible kmer of length k in the window w
                candidateMinmer = searchString[i+j:i+j+self.k]
                if not minmer or candidateMinmer < minmer:
                    minmer = candidateMinmer
                    sites = typed.List([int(i+j)])
                    # print(typeof(self.minimizerMap.get(minmer)))
                    # print(len(self.minimizerMap.get(minmer)))
#                    yield (sites[0], self.minimizerMap[minmer])
#            except Exception as e: 
#                print(e)
#            print("getMatches()", sites)
            if sites[0] not in yieldedSites and minmer in self.minimizerMap:# self.minimizerMap.get(minmer):
                yieldedSites.append(sites[0])
                yield (sites[0], self.minimizerMap[minmer])

class SeedCluster:
    """ Represents a set of seeds between two strings.
    """
    def __init__(self, seeds):
        """ Seeds is a list of pairs [ (x_1, y_1), (x_2, y_2), ..., ], each is an instance of a seed 
        (see static cluster seeds method below: static methods: 
            https://realpython.com/blog/python/instance-class-and-static-methods-demystified/)
            
        My bug with this constructor:
            Traceback (most recent call last):
                File "simpleMap.py", line 451, in <module>
                main()
                File "simpleMap.py", line 436, in main
                alignment = simpleMap(targetString, minimizerIndex, query.sequence.upper(), config)
                File "simpleMap.py", line 352, in simpleMap
                mapForwards(queryString)
                File "simpleMap.py", line 319, in mapForwards
                for seedCluster in SeedCluster.clusterSeeds(list(seeds), l=config.l):
                    File "simpleMap.py", line 260, in clusterSeeds
                    seedClusterSet.add(SeedCluster(nodeList))
                    File "simpleMap.py", line 206, in __init__
                    self.minY = min(ys)
                    TypeError: <lambda>() missing 1 required positional argument: 'y'
                    
        But here is my seed: clusterSeeds (log): nodeList before SeedCluster(nodeList) [(0, 2099931)]
        """
        seeds = list(seeds)
        # print("SeedCluster init()", list(seeds)) # might require yield
        seeds.sort()
        self.seeds = seeds
        # Gather the minimum and maximum x and y coordinates
        self.minX = seeds[0][0]
        self.maxX = seeds[-1][0]
        ## ys = map(lambda (x, y) : y, seeds)
        ys = list(map(lambda xy : xy[1], seeds))
        # self.minY = min(ys)
        # self.maxY = max(ys)
        self.minY = min(list(ys))
        self.maxY = max(list(ys))

    @staticmethod
    def clusterSeeds(seeds, l):
        """ Cluster seeds (k-mer instances) in two strings. This is a static constructor 
        method that creates a set of SeedCluster instances.
        
        Here seeds is a list of tuples, each tuple has the form (x, (y_1, y_2, ... )), 
        where x is the coordinate in the first string and y_1, y_2, ... are coordinates 
        in the second string. Each pair of x and y_i is an occurence of a shared k-mer 
        in both strings, termed a *seed*, such that the k-mer occurrence starts at 
        position x in the first string and starts at position y_i in the second string.
        The input seeds list contains no duplicates and is sorted in ascending order, 
        first by x coordinate (so each successive tuple will have a greater  
        x coordinate), and then each in tuple the y coordinates are sorted in ascending order.
        
        Two seeds (x_1, y_1), (x_2, y_2) are *close* if the absolute distances
        | x_2 - x_1 | and | y_2 - y_1 | are both less than or equal to l.   
        
        Consider a *seed graph* in which the nodes are the seeds, and there is an edge 
        between two seeds if they
        are close. clusterSeeds returns the connected components of this graph
        (https://en.wikipedia.org/wiki/Connected_component_(graph_theory)).
        
        The return value is a Python set of SeedCluster object, each representing a 
        connected component of seeds in the seed graph.
        """ 
        
        # Code to complete - you are free to define other functions as you like
        seedClusterSet = set()
        nodeList = list()
        orphanNodeList = list()
        for i in seeds:
            # Performs an optimized O(n) DFS because we know that the minmers are left-to-right sorted
            x1 = i[0]
            # First iterate all of the seeds of the form (x, (y_1, y_2, ... ))
            for y1j in i[1]:
                # Then iterate all of our string2 list of y coords
                if not len(nodeList):
                    # If our current connected component is empty, then start a new one
                    nodeList.append((x1, y1j))                    
                elif abs(x1 - nodeList[-1][0]) <= l and abs(y1j - nodeList[-1][1]) <= l:
                    # If we are connected with our current component, then append to it
                    nodeList.append((x1, y1j))
                else:
                    orphanNodeList.append((x1, y1j))
        # Try one last time to attach orphans
        connectOrphans = False
        if len(orphanNodeList):
            connectOrphans = True
        found = False
        while connectOrphans:
            # Iterate the orphans and if we add any then loop again
            orphanLength = len(orphanNodeList)
            # for x1, y1j in orphanNodeList:
            for o in range(orphanLength):
                x1 = orphanNodeList[o][0]
                y1j = orphanNodeList[o][1]
                for node in nodeList[:]: # because we already checked the last one...
                    if abs(x1 - node[0]) <= l and abs(y1j - node[1]) <= l:
                        # If we are connected with our current component, then append to it
                        nodeList.append((x1, y1j))
                        del orphanNodeList[o]
                        found = True
                        break
                        # found = True
                        # If we connect then we can break
                if found:
                    found = False
                    break
            if len(orphanNodeList) == orphanLength:
                connectOrphans = False

        # Dump the nodeList if everything so far was connected
        if len(nodeList):
            seedClusterSet.add(SeedCluster(nodeList))

        return seedClusterSet

class SmithWaterman(object):
    def __init__(self, string1, string2, gapScore=-2, matchScore=3, mismatchScore=-3):
        """ Finds an optimal local alignment of two strings.
        
        Implements the Smith-Waterman algorithm: 
        https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
        """
        # Code to complete to compute the edit matrix
        self.editMatrix = numpy.zeros(shape=[len(string1)+1, len(string2)+1], dtype=int) # Numpy matrix representing edit matrix
        # Preinitialized to have zero values
        self.backPointMatrix = [[0 for j in range(len(string2)+1)] for i in range(len(string1)+1)]

        # Now complete the rest of the matrix
        for i in range(1, len(self.editMatrix[:,0])):
          # Iterate each row in the matrix, ie each character in string1
          for j in range(1, len(self.editMatrix[0,:])):
            # Then iterate each column in the matrix, ie each character in string2
            cornerScore, scoreUp, scoreLeft = 0, 0, 0
            if string1[i-1] == string2[j-1]:
              cornerScore = self.editMatrix[i-1, j-1] + matchScore
            else:
              cornerScore = self.editMatrix[i-1, j-1] + mismatchScore
            scoreUp = self.editMatrix[i-1][j] + gapScore
            scoreLeft = self.editMatrix[i][j-1] + gapScore
            self.editMatrix[i,j] = max(cornerScore, scoreUp, scoreLeft, 0)
            best = numpy.argmax(numpy.asarray([cornerScore, scoreUp, scoreLeft]))
            if not best:
                self.backPointMatrix[i][j] = [i-1, j-1]
            elif best == 1:
                self.backPointMatrix[i][j] = [i-1, j]
            else:
                self.backPointMatrix[i][j] = [i, j-1]
            
    def getAlignment(self):
        """ Returns an optimal local alignment of two strings. Alignment is returned as 
        an ordered list of aligned pairs.
        
        e.g. For the two strings GATTACA and CTACC an optimal local alignment
        is (GAT)TAC(A)
             (C)TAC(C)
        where the characters in brackets are unaligned. This alignment would be returned as
        [ (3, 1), (4, 2), (5, 3) ] 
        
        In cases where there is a tie between optimal sub-alignments use the following rule:
        Let (i, j) be a point in the edit matrix, if there is a tie between possible sub-alignments
        (e.g. you could chooose equally between different possibilities), choose the (i, j) to (i-1, j-1)
        (match) in preference, then the (i, j) to (i-1, j) (insert in string1) in preference and
        then (i, j) to (i, j-1) (insert in string2).
        """
        # Code to complete - generated by traceback through matrix to generate aligned pairs
        alignedPairs = []

        # CITATION: https://stackoverflow.com/questions/9482550/argmax-of-numpy-array-returning-non-flat-indices
        bestScoringIndex = numpy.unravel_index(numpy.argmax(self.editMatrix), 
                                               (len(self.editMatrix), len(self.editMatrix[0])))
        currScore = self.editMatrix[bestScoringIndex[0], bestScoringIndex[1]]
        i = bestScoringIndex[0]
        j = bestScoringIndex[1]
        while i>0 or j>0:
            try:
                currElement = self.backPointMatrix[i][j]
                i = currElement[0]
                j = currElement[1]
                if self.editMatrix[i][j] < currScore:
                    alignedPairs.append((i,j))
                    currScore = self.editMatrix[i][j]
            except:
                break
        
        return sorted(alignedPairs)
    
    def getMaxAlignmentScore(self):
        """ Returns the maximum alignment score
        
        This is different from Needleman-Wunsch because the high score can be anywhere
        in the matrix!
        """
        # Code to complete
        return numpy.amax(self.editMatrix)

def simpleMap(targetString, minimizerIndex, queryString, config):
    """ Function takes a target string with precomputed minimizer index and a query string
    and returns the best alignment it finds between target and query, using the given options specified in config.
    
    Maps the string in both its forward and reverse complement orientations.    
    """
    bestAlignment = [None]

    def mapForwards(queryString): # , targetString, bestAlignment, minimizerIndex, config):
        """ Maps the query string forwards
        """
        # Find seed matches, aka "aligned kmers"
        # print("mapForwards (log):", list(minimizerIndex.getMatches(queryString)))
        startTime = time.time()
        # print('Calling getMatches() with ', queryString[:20])
        # seeds = list(minimizerIndex.getMatches(queryString))
        try:
            seeds = list(minimizerIndex.getMatches(queryString))
        except:
            print("No generator for", queryString[:20])
            seeds = list()
        print("Time to getMatches()", time.time() - startTime)
        
        # For each cluster of seeds
        startTime = time.time()
        for seedCluster in SeedCluster.clusterSeeds(seeds, l=config.l):
            print("Time to clusterSeeds()", time.time() - startTime)
            
            # Get substring of query and target to align
            queryStringStart = max(0, seedCluster.minX - config.c) # Inclusive coordinate
            queryStringEnd = min(len(queryString), seedCluster.maxX + config.k + config.c) # Exclusive coordinate
            querySubstring = queryString[queryStringStart:queryStringEnd]
            
            targetStringStart = max(0, seedCluster.minY - config.c) # Inclusive coordinate
            targetStringEnd = min(len(targetString), seedCluster.maxY + config.k + config.c) # Exclusive coordinate
            targetSubstring = targetString[targetStringStart:targetStringEnd]
            
            # Align the genome and read substring
            startTime = time.time()
            alignment = SmithWaterman(targetSubstring, querySubstring, 
                                      gapScore=config.gapScore, 
                                      matchScore=config.matchScore,
                                      mismatchScore=config.mismatchScore)
            print("Time to SmithWaterman()", time.time()-startTime)
            
            # Update best alignment if needed
            if bestAlignment[0] == None or alignment.getMaxAlignmentScore() > bestAlignment[0].getMaxAlignmentScore():
                bestAlignment[0] = alignment
        
        return bestAlignment
    
    def reverseComplement(string):
        """Computes the reverse complement of a string
        """
        rMap = { "A":"T", "T":"A", "C":"G", "G":"C", "N":"N"}
        return "".join(rMap[i] for i in string[::-1])
    
    # Run mapping forwards and reverse
    mapForwards(queryString)
    mapForwards(reverseComplement(queryString))
    return bestAlignment[0]

class Config():
    """ Minimal configuration class for handing around parameters
    """
    def __init__(self):
        self.w = 30
        self.k = 20
        self.t = 10
        self.l = 30
        self.c = 100
        self.gapScore=-2
        self.matchScore=3
        self.mismatchScore=-3
        self.logLevel = "INFO"
        
def main():
    # Read parameters
    config = Config()
    
    #Parse the inputs args/options
    parser = argparse.ArgumentParser(usage="target_fasta query_fastq [options]") # , version="%prog 0.1")

    parser.add_argument("target_fasta", type=str,
                        help="The target genome fasta file.")
    parser.add_argument("query_fastq", type=str,
                        help="The query sequences.")
    
    parser.add_argument("--w", dest="w", type=int, help="Length of minimizer window. Default=%s" % config.w, default=config.w)
    parser.add_argument("--k", dest="k", type=int, help="Length of k-mer. Default=%s" % config.k, default=config.k)
    parser.add_argument("--t", dest="t", type=int, help="Discard minmers that occur more frequently " 
                                            "in the target than t. Default=%s" % config.t, default=config.t)
    parser.add_argument("--l", dest="l", type=int, help="Cluster two minmers into the same cluster if within l bases of"
                                            " each other in both target and query. Default=%s" % config.l, default=config.l)
    parser.add_argument("--c", dest="c", type=int, help="Add this many bases to the prefix and suffix of a seed cluster in the"
                                            " target and query sequence. Default=%s" % config.c, default=config.c)
    parser.add_argument("--gapScore", type=float, dest="gapScore", help="Smith-Waterman gap-score. Default=%s" % 
                      config.gapScore, default=config.gapScore)
    parser.add_argument("--matchScore", type=float, dest="matchScore", help="Smith-Waterman match-score. Default=%s" % 
                      config.gapScore, default=config.gapScore)
    parser.add_argument("--mismatchScore", type=float, dest="mismatchScore", help="Smith-Waterman mismatch-score. Default=%s" % 
                      config.mismatchScore, default=config.mismatchScore)
    parser.add_argument("--log", dest="logLevel", help="Logging level. Default=%s" % 
                      config.logLevel, default=config.logLevel)
    
    options = parser.parse_args()
    
    # Parse the log level
    numeric_level = getattr(logging, options.logLevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.logLevel)
    
    # Setup a logger
    logger.setLevel(numeric_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.debug("Established logger")
    
    startTime = time.time()
    
    # Parse the target sequence and read the first sequence
    with pysam.FastaFile(options.target_fasta) as targetFasta:
        targetString = targetFasta.fetch(targetFasta.references[0])
        logger.info("Parsed target string. Length: %s in %d seconds." % (len(targetString), time.time()-startTime))
    
    # Build minimizer index
    minimizerIndex = MinimizerIndexer(targetString.upper(), w=options.w, k=options.k, t=options.t)
    print("minimizerMap keys", list(minimizerIndex.minimizerMap.keys())[:20],
          "minmerOccurrences keys", list(minimizerIndex.minmerOccurrences.keys())[:20])
    minmerInstances = sum(map(len, minimizerIndex.minimizerMap.values()))
    logger.info("Built minimizer index in %s seconds. #minmers: %s, #minmer instances: %s" %
                 ((time.time()-startTime), len(minimizerIndex.minimizerMap), minmerInstances))
    
    # Open the query files
    alignmentScores = [] # Array storing the alignment scores found
    with pysam.FastqFile(options.query_fastq) as queryFastq:
        # For each query string build alignment
        for query, queryIndex in zip(queryFastq, range(sys.maxsize)): # xrange(sys.maxint)):
            ## print queryIndex
            print(queryIndex)
            alignment = simpleMap(targetString, minimizerIndex, query.sequence.upper(), config)
            alignmentScore = 0 if alignment is None else alignment.getMaxAlignmentScore()
            alignmentScores.append(alignmentScore)
            logger.info("Mapped query sequence #%i, length: %s alignment_found?: %s "
                        "max_alignment_score: %s" % 
                        (queryIndex, len(query.sequence), alignment is not None, alignmentScore)) 
    # Print some stats
    logger.info("Finished alignments in %s total seconds, average alignment score: %s" % 
                    (time.time()-startTime, float(sum(alignmentScores))/len(alignmentScores)))
if __name__ == '__main__':
    main()
