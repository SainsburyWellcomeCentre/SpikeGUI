OVERVIEW
--------

This package is designed to handle all analysis required for trials-based calcium imaging and probe
recordings. The initial implementation was heavily focused on the analysis of experiments in which
the stimulus is a rotational waveform and the acquisition method was two photon calcium imaging. This has
since been extended to handle probe data for the same experiments and aims to follow these principles:

* analyses common to both are not repeated - there is a shared pipeline that only deviates when required.
* ideally code only needs to be modified in one place
* probe analysis/calcium analysis are simply subclassed and extended for specific requirements


STRUCTURE
---------

The package is split into several abstractions as follows:

* Stimulus - this defines the starts and ends of the whole stimulus and is split into the baseline and all of the individually analysable sections of the stimulus. For rotations this is automatically defined based on the waveform. Some aspects must currently be defined by the user(not implemented).
* EventsCollection - this is a group of events. In calcium data events can be analysed in multiple ways (peak, integral, duration frequency etc) so this class contains some of the logic required.
* Trial - in contrast to the stimulus, the trial knows about the data and it has a stimulus, which is uses as a view onto the data. Trials have events, which are all relative to the onset of the trial. The trial doesn't know anything about events that occur outside its start and end.
* Block - trials are grouped into a block of trials, but they do not need to be contiguous. We use a context manager to selectively flag the trials that will be included in analysis. This happens at the level of the block.
* Cell - each cell has a block of trials
* FieldOfView - the field of view (FOV) is a group of cells. The user is supposed to use functions at this level only. A FOV is given a list of cell ids, which determines which cells go into the pipeline. FOV functions typically take a dictionary of conditions that is used to filter trials in the block while running some analytical function.


Calcium imaging specifics
=========================

- Calcium imaging data also includes event detection and processing. There is a GUI specifically for manual curation of traces.

Probe specifics
===============

The probe package inherits directly from this code (so there is a ProbeStimulus, ProbeTrial etc). Additionally
it has its own io modules which handle data that is specific to the probe only. Everything else is pretty close
to the parent class.

The probe code has to deal with *within-stimulus conditions* and *across stimuli conditions*

*Within stimulus condition* refers to the structure of the stimulus (i.e. baseline vs. visual stimulus, or baseline vs
clockwise or counterclockwise. *Across stimuli conditions* are parameters that the user wants to compare between different
trials. For example, lets say you have 5 trials - trials 1, 3 and 4 might have been done in the dark, while 2 and 5
might have been done in the light.

To give the user full flexibility to mix and match any combination of within and across conditions, we generate a
pseudo-database from the field of view, which takes a relatively long time and is stored on disk. this can then be queried
to filter trials to those that the user wants to compare.
