	??T????@??T????@!??T????@	?w??????w?????!?w?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??T????@^??v1??A??V????@Y?P??dV??*	/?$	z@2F
Iterator::Model?2??(??!=(K??S@)QۆQ<??1?kʳ??R@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatE?u?????!?S?%9&@)1???Cޢ?1?Ae}?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice<?D???![???U@)<?D???1[???U@:Preprocessing2U
Iterator::Model::ParallelMapV2^???j???!O(???@)^???j???1O(???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate ?O????!'?u}/?@)??F????1C?>9-?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorQ??C?R??!?/FX?@)Q??C?R??1?/FX?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????!?_Ӗ?4@)Um7?7M?1H#?A>Z??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??2nj??!?Bk??@)?@?C?b?1&?G????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?w?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	^??v1??^??v1??!^??v1??      ??!       "      ??!       *      ??!       2	??V????@??V????@!??V????@:      ??!       B      ??!       J	?P??dV???P??dV??!?P??dV??R      ??!       Z	?P??dV???P??dV??!?P??dV??JCPU_ONLYY?w?????b 