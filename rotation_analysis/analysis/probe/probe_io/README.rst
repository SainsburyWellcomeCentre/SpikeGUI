The IO classes in this directory are specifically intended to be used for handling triggers: stimulus onsets that arise
from different parts of the acquisition set up. There are three sources of information, and therefore three different
IO classes:

* TriggerTraceIo deals with the 'real time' raw triggers on the probe (recorded as a TTL), that give the actual onset of the stimulus
* IgorIo deals with the stimulus waveforms themselves - i.e. how is the stimulus broken into different sections
* BonsaiIO deals with all metadata surrounding a given stimulus - i.e. the parameters of stimuli presented by bonsai

These all operate in the same way - an ordered list of stimulus related information that is accessed by trigger index.
Index n (TriggerTraceIo) == idx n (BonsaiIo) == idx n IgorIo
