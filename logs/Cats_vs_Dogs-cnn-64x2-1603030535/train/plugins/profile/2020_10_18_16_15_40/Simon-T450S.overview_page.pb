?	?/?^?@?@?/?^?@?@!?/?^?@?@	??>jom???>jom?!??>jom?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?/?^?@?@>v()???A?f+@?@Y)?k{???*	?z?G?W@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat
??t??!???)P@@)??:7mƙ?1e??օ?:@:Preprocessing2F
Iterator::Model??|?r٠?!????RzA@)??!S>??1P??4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?"?Ƥ??!73?qsi0@)?"?Ƥ??173?qsi0@:Preprocessing2U
Iterator::Model::ParallelMapV2)?'?$???!%??J?+@))?'?$???1%??J?+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????·??!?]x?xh8@)]???~?1??,??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipv??2SZ??!?????BP@)???fw?1?D?%?E@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorfI??Z?v?!	??	5?@)fI??Z?v?1	??	5?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ؖg??!???'?Y:@)??0?]?1?q?%??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??>jom?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>v()???>v()???!>v()???      ??!       "      ??!       *      ??!       2	?f+@?@?f+@?@!?f+@?@:      ??!       B      ??!       J	)?k{???)?k{???!)?k{???R      ??!       Z	)?k{???)?k{???!)?k{???JCPU_ONLYY??>jom?b Y      Y@q??GU?o?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 