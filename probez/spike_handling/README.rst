The spike_handling modules are for holding the output data of KiloSort, and for basic handling of raw traces
such as getting the waveforms for each event.
It includes:

* SpikeIo - this class is a container for the KiloSort data used to feed data to other classes, or to used directly to perform basic exploration of probe data.

* Cluster - an abstraction for dealing with all forms of data relevant to a specific cluster. This includes the location of the cluster on the probe, quality measures and basic plotting

* Waveforms - generic functions for extracting waveforms - used by both SpikeIo and Cluster
