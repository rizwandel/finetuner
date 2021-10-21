Search.setIndex({docnames:["api/finetuner","api/finetuner.helper","api/finetuner.labeler","api/finetuner.labeler.executor","api/finetuner.tailor","api/finetuner.tailor.base","api/finetuner.tailor.keras","api/finetuner.tailor.paddle","api/finetuner.tailor.pytorch","api/finetuner.toydata","api/finetuner.tuner","api/finetuner.tuner.base","api/finetuner.tuner.dataset","api/finetuner.tuner.dataset.helper","api/finetuner.tuner.keras","api/finetuner.tuner.keras.datasets","api/finetuner.tuner.keras.losses","api/finetuner.tuner.paddle","api/finetuner.tuner.paddle.datasets","api/finetuner.tuner.paddle.losses","api/finetuner.tuner.pytorch","api/finetuner.tuner.pytorch.datasets","api/finetuner.tuner.pytorch.losses","api/finetuner.tuner.summary","api/modules","basics/data-format","basics/fit","basics/glossary","basics/index","components/index","components/labeler","components/overview","components/tailor","components/tuner","get-started/celeba","get-started/covid-qa","get-started/fashion-mnist","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/finetuner.rst","api/finetuner.helper.rst","api/finetuner.labeler.rst","api/finetuner.labeler.executor.rst","api/finetuner.tailor.rst","api/finetuner.tailor.base.rst","api/finetuner.tailor.keras.rst","api/finetuner.tailor.paddle.rst","api/finetuner.tailor.pytorch.rst","api/finetuner.toydata.rst","api/finetuner.tuner.rst","api/finetuner.tuner.base.rst","api/finetuner.tuner.dataset.rst","api/finetuner.tuner.dataset.helper.rst","api/finetuner.tuner.keras.rst","api/finetuner.tuner.keras.datasets.rst","api/finetuner.tuner.keras.losses.rst","api/finetuner.tuner.paddle.rst","api/finetuner.tuner.paddle.datasets.rst","api/finetuner.tuner.paddle.losses.rst","api/finetuner.tuner.pytorch.rst","api/finetuner.tuner.pytorch.datasets.rst","api/finetuner.tuner.pytorch.losses.rst","api/finetuner.tuner.summary.rst","api/modules.rst","basics/data-format.md","basics/fit.md","basics/glossary.md","basics/index.md","components/index.md","components/labeler.md","components/overview.md","components/tailor.md","components/tuner.md","get-started/celeba.md","get-started/covid-qa.md","get-started/fashion-mnist.md","index.md"],objects:{"":{finetuner:[0,0,0,"-"]},"finetuner.helper":{AnyDNN:[1,2,1,""],AnyDataLoader:[1,2,1,""],AnyOptimizer:[1,2,1,""],DocumentArrayLike:[1,2,1,""],DocumentSequence:[1,2,1,""],LayerInfoType:[1,2,1,""],get_framework:[1,1,1,""],is_seq_int:[1,1,1,""]},"finetuner.labeler":{executor:[3,0,0,"-"],fit:[2,1,1,""]},"finetuner.labeler.executor":{DataIterator:[3,3,1,""],FTExecutor:[3,3,1,""]},"finetuner.labeler.executor.DataIterator":{add_fit_data:[3,4,1,""],requests:[3,5,1,""],store_data:[3,4,1,""],take_batch:[3,4,1,""]},"finetuner.labeler.executor.FTExecutor":{embed:[3,4,1,""],fit:[3,4,1,""],get_embed_model:[3,4,1,""],requests:[3,5,1,""],save:[3,4,1,""]},"finetuner.tailor":{base:[5,0,0,"-"],display:[4,1,1,""],keras:[6,0,0,"-"],paddle:[7,0,0,"-"],pytorch:[8,0,0,"-"],to_embedding_model:[4,1,1,""]},"finetuner.tailor.base":{BaseTailor:[5,3,1,""]},"finetuner.tailor.base.BaseTailor":{display:[5,4,1,""],embedding_layers:[5,6,1,""],summary:[5,4,1,""],to_embedding_model:[5,4,1,""]},"finetuner.tailor.keras":{KerasTailor:[6,3,1,""]},"finetuner.tailor.keras.KerasTailor":{summary:[6,4,1,""],to_embedding_model:[6,4,1,""]},"finetuner.tailor.paddle":{PaddleTailor:[7,3,1,""]},"finetuner.tailor.paddle.PaddleTailor":{summary:[7,4,1,""],to_embedding_model:[7,4,1,""]},"finetuner.tailor.pytorch":{PytorchTailor:[8,3,1,""]},"finetuner.tailor.pytorch.PytorchTailor":{summary:[8,4,1,""],to_embedding_model:[8,4,1,""]},"finetuner.toydata":{generate_fashion_match:[9,1,1,""],generate_qa_match:[9,1,1,""]},"finetuner.tuner":{base:[11,0,0,"-"],dataset:[12,0,0,"-"],fit:[10,1,1,""],keras:[14,0,0,"-"],paddle:[17,0,0,"-"],pytorch:[20,0,0,"-"],save:[10,1,1,""],summary:[23,0,0,"-"]},"finetuner.tuner.base":{BaseDataset:[11,3,1,""],BaseLoss:[11,3,1,""],BaseTuner:[11,3,1,""]},"finetuner.tuner.base.BaseLoss":{arity:[11,5,1,""]},"finetuner.tuner.base.BaseTuner":{arity:[11,6,1,""],embed_model:[11,6,1,""],fit:[11,4,1,""],save:[11,4,1,""]},"finetuner.tuner.dataset":{SiameseMixin:[12,3,1,""],TripletMixin:[12,3,1,""],helper:[13,0,0,"-"]},"finetuner.tuner.dataset.helper":{get_dataset:[13,1,1,""]},"finetuner.tuner.keras":{KerasTuner:[14,3,1,""],datasets:[15,0,0,"-"],losses:[16,0,0,"-"]},"finetuner.tuner.keras.KerasTuner":{fit:[14,4,1,""],save:[14,4,1,""]},"finetuner.tuner.keras.datasets":{SiameseDataset:[15,3,1,""],TripletDataset:[15,3,1,""]},"finetuner.tuner.keras.losses":{CosineSiameseLoss:[16,3,1,""],CosineTripletLoss:[16,3,1,""],EuclideanSiameseLoss:[16,3,1,""],EuclideanTripletLoss:[16,3,1,""]},"finetuner.tuner.keras.losses.CosineSiameseLoss":{arity:[16,5,1,""],call:[16,4,1,""]},"finetuner.tuner.keras.losses.CosineTripletLoss":{arity:[16,5,1,""],call:[16,4,1,""]},"finetuner.tuner.keras.losses.EuclideanSiameseLoss":{arity:[16,5,1,""],call:[16,4,1,""]},"finetuner.tuner.keras.losses.EuclideanTripletLoss":{arity:[16,5,1,""],call:[16,4,1,""]},"finetuner.tuner.paddle":{PaddleTuner:[17,3,1,""],datasets:[18,0,0,"-"],losses:[19,0,0,"-"]},"finetuner.tuner.paddle.PaddleTuner":{fit:[17,4,1,""],save:[17,4,1,""]},"finetuner.tuner.paddle.datasets":{SiameseDataset:[18,3,1,""],TripletDataset:[18,3,1,""]},"finetuner.tuner.paddle.losses":{CosineSiameseLoss:[19,3,1,""],CosineTripletLoss:[19,3,1,""],EuclideanSiameseLoss:[19,3,1,""],EuclideanTripletLoss:[19,3,1,""]},"finetuner.tuner.paddle.losses.CosineSiameseLoss":{arity:[19,5,1,""],forward:[19,4,1,""]},"finetuner.tuner.paddle.losses.CosineTripletLoss":{arity:[19,5,1,""],forward:[19,4,1,""]},"finetuner.tuner.paddle.losses.EuclideanSiameseLoss":{arity:[19,5,1,""],forward:[19,4,1,""]},"finetuner.tuner.paddle.losses.EuclideanTripletLoss":{arity:[19,5,1,""],forward:[19,4,1,""]},"finetuner.tuner.pytorch":{PytorchTuner:[20,3,1,""],datasets:[21,0,0,"-"],losses:[22,0,0,"-"]},"finetuner.tuner.pytorch.PytorchTuner":{fit:[20,4,1,""],save:[20,4,1,""]},"finetuner.tuner.pytorch.datasets":{SiameseDataset:[21,3,1,""],TripletDataset:[21,3,1,""]},"finetuner.tuner.pytorch.losses":{CosineSiameseLoss:[22,3,1,""],CosineTripletLoss:[22,3,1,""],EuclideanSiameseLoss:[22,3,1,""],EuclideanTripletLoss:[22,3,1,""]},"finetuner.tuner.pytorch.losses.CosineSiameseLoss":{arity:[22,5,1,""],forward:[22,4,1,""]},"finetuner.tuner.pytorch.losses.CosineTripletLoss":{forward:[22,4,1,""]},"finetuner.tuner.pytorch.losses.EuclideanSiameseLoss":{arity:[22,5,1,""],forward:[22,4,1,""]},"finetuner.tuner.pytorch.losses.EuclideanTripletLoss":{arity:[22,5,1,""],forward:[22,4,1,""]},"finetuner.tuner.summary":{NumericType:[23,2,1,""],ScalarSummary:[23,3,1,""],SummaryCollection:[23,3,1,""]},"finetuner.tuner.summary.ScalarSummary":{floats:[23,4,1,""]},"finetuner.tuner.summary.SummaryCollection":{dict:[23,4,1,""],save:[23,4,1,""]},finetuner:{fit:[0,1,1,""],helper:[1,0,0,"-"],labeler:[2,0,0,"-"],tailor:[4,0,0,"-"],toydata:[9,0,0,"-"],tuner:[10,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","data","Python data"],"3":["py","class","Python class"],"4":["py","method","Python method"],"5":["py","attribute","Python attribute"],"6":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:function","2":"py:data","3":"py:class","4":"py:method","5":"py:attribute","6":"py:property"},terms:{"0":[0,9,10,14,16,17,19,20,22,25,30,32,33,34,36,37],"00":[30,34,36],"000":25,"00010502179":25,"001":[0,10,14,17,20],"002585097":25,"011804931":25,"028714137":25,"03":34,"06":34,"08":[10,14,17,20],"0e7ec5aa":25,"0e7ec7c6":25,"0e7ecd52":25,"0e7ece7":25,"1":[9,16,19,22,26,30,32,33,34,35,36,37],"10":[0,10,11,14,17,20,25,33],"100":[9,26,30,32,34,35,37],"1000":[32,34],"100480":32,"102764544":32,"109":[30,34,36],"11":34,"112":32,"1180160":32,"11ec":[25,35,36],"128":[32,33,36,37],"12900":32,"132":[30,34,36],"135":[30,34,36],"14":32,"141":36,"147584":32,"16781312":32,"172":[30,34,36],"1792":32,"18":[30,34,36],"180":34,"19":36,"1bab":36,"1bae":35,"1e":[10,14,17,20],"1e008a366d49":[25,35,36],"1f9f":25,"1faa":25,"2":[9,11,16,19,22,25,26,32,33,35,36,37],"20":34,"224":[32,34],"22900":36,"231":[30,34,36],"2359808":32,"25088":32,"2508900":32,"256":[0,10,11,14,17,20,32],"28":[32,33,36,37],"28x28":25,"295168":32,"29672":30,"3":[9,11,16,19,22,25,26,32,34,36,37],"31":36,"32":[32,33,35,36,37],"320000":32,"33":36,"3300":32,"36928":32,"3gb":34,"4":[25,26,34,37],"4096":32,"409700":32,"4097000":32,"4128":32,"481":[9,25],"49":36,"5":[25,33,37],"5000":[32,33,35],"512":32,"52621":36,"53":34,"56":[9,32,36],"5716974480":36,"5794172560":35,"590080":32,"6":[25,34],"60":25,"60000":9,"61130":30,"61622":34,"64":[32,33,35],"65":36,"66048":32,"6620":34,"66560":32,"67":36,"67432a92":25,"67432cd6":25,"685":[30,34],"7":[32,36,37],"73856":32,"75":34,"784":32,"784x128x32":33,"9":[10,14,17,20,36,37],"94":[30,34,36],"99":[10,14,17,20],"999":[10,14,17,20],"9a49":25,"abstract":[3,5,11],"case":37,"class":[3,5,6,7,8,11,12,14,15,16,17,18,19,20,21,22,23,25,32,33,35,36],"default":33,"do":[25,26,32,34,37],"final":[16,19,22,32,34,35,36],"float":[0,10,14,17,20,23,25],"function":[1,3,11,14,17,20,25,32,33],"import":[1,25,26,30,32,33,34,35,36,37],"int":[0,5,6,7,8,9,10,11,14,17,20,23,32],"new":[30,32,33,37],"public":[30,34,36,37],"return":[0,1,2,4,5,6,7,8,9,10,11,14,17,19,20,22,23,32,33,35],"switch":[33,37],"true":[0,1,9,25,26,30,32,33,34,35,36,37],"while":30,A:[19,22,27,30],But:[25,26,32,37],By:33,For:[9,11,25,27,30,32,34],If:[9,25,30,34,35],In:[25,30,32,33,34,35,36,37],It:[1,5,6,7,8,9,25,30,32,33,35,37],Its:33,No:[26,37],On:34,One:9,The:[1,5,6,7,8,9,10,14,16,17,19,20,22,23,25,30,31,33,34,35,36,37],Then:25,To:[5,6,7,8,25,32,33],_:[32,33,35],__init__:7,__module__:1,_i:33,_j:33,_n:33,_p:33,a207:36,a46a:25,a5dd3158:25,a5dd3784:25,a5dd3b94:25,a5dd3d74:25,aaaaaaaa:25,aaaaaaaaaaaaaa:36,aaaaaaaaaaaaaaaa:35,aaaaaaaaekaaaaaaaaawqaaaaaaaabpa:25,abc:[5,11],abl:[30,34],about:[26,33,37],abov:[30,34],ac8a:25,accept:[25,33,34,35,36],access:[30,34,36],accompani:37,accord:33,accur:32,accuraci:36,achiev:32,action:32,activ:[30,31,32,33,34,36,37],actual:1,ad:32,adam:[0,10,14,17,20],adaptiveavgpool2d_32:32,add:[25,32],add_fit_data:3,addit:[3,32],address:[30,34,36],adjac:25,adjust:30,advanc:25,affect:30,after:[5,6,7,8,25,30,32,33,34,35,36],afterward:[30,32],again:32,agnost:37,ai:37,algorithm:30,alia:[1,23],all:[5,6,7,8,9,16,19,22,23,26,30,32,33,34,37],allow:[11,25,37],alreadi:[26,30,37],also:[16,19,22,30,32,33,35],although:33,alwai:[3,25,30],an:[1,5,6,7,8,10,26,27,30,31,32,33,37],anchor:[16,19,22],ani:[1,5,6,7,8,26,27,30,31,32,35,36,37],answer:[25,26,30,37],anydataload:1,anydnn:[0,1,4,5,6,7,8,10,11,14,17,20,32],anyoptim:1,anyth:25,apach:37,apart:32,api:[25,32,37],app:37,append:25,applic:[32,34,37],aqaaaaaaaaacaaaaaaaaaamaaaaaaaaa:25,ar:[3,5,6,7,8,10,11,14,17,20,25,26,30,32,33,34,35,36,37],ara:3,architectur:[5,6,7,8,32],arg:[5,10,11,14,16,17,19,20,22],argument:[3,10,14,17,20,30,34],ariti:[11,13,16,19,22],arrai:[1,25,34,35,36],ask:[27,30],async:30,auf:25,auto:3,avail:[5,6,7,8,11,14,17,20,30,34,36],averag:[16,19,22],avoid:3,axi:9,b32d:35,b9557788:36,b:[9,25,27],baaaaaaaaaafaaaaaaaaaayaaaaaaaaa:25,back:37,backend:[30,34,36,37],bad:[25,30],bar:[30,35],base64:36,base:[0,1,3,4,6,7,8,10,12,14,15,16,17,18,19,20,21,22,23,30,34,35,36],basedataset:[11,15,18,21],baseexecutor:3,baseloss:[11,16,19,22],basetailor:[5,6,7,8],basetun:[11,14,17,20],batch:[5,6,7,8,10,14,16,17,19,20,22,27],batch_first:[32,33,35],batch_siz:[0,10,11,14,17,20],becaus:[25,30],been:[30,32,35],befor:[34,35,36],behav:30,being:[16,19,22],belong:[16,19,22],below:[25,26,30,33,37],besid:30,best:34,beta_1:[10,14,17,20],beta_2:[10,14,17,20],better:[11,16,19,22,27,30,31,34,35,36,37],between:[16,19,22,25,33,37],bewar:34,bidirect:[32,35],big:33,bigger:9,blob:[25,35,36],block1_conv1:32,block1_conv2:32,block1_pool:32,block2_conv1:32,block2_conv2:32,block2_pool:32,block3_conv1:32,block3_conv2:32,block3_conv3:32,block3_pool:32,block4_conv1:32,block4_conv2:32,block4_conv3:32,block4_pool:32,block5_conv1:32,block5_conv2:32,block5_conv3:32,block5_pool:32,bonjour:25,bool:[0,1,5,6,7,8,9,32],both:[16,19,22],bottleneck:32,browser:[30,34],buffer:[25,35,36],build:[25,32,33,37],built:[25,33],butteri:37,button:30,c:[5,6,7,8,9],calendar:37,call:[11,16,25,32,33],callabl:[1,10,14,17,20],can:[5,6,7,8,9,25,26,30,31,32,33,34,35,36,37],cannot:25,card:30,center:[10,14,17,20],certain:32,chang:[11,30,34],channel:[9,25,37],channel_axi:9,chapter:25,chat:37,chatbot:35,check:[5,6,7,8],checkout:[35,36],choic:30,choos:34,chop:32,classif:27,clear_labels_on_start:[0,2,3],click:30,co:33,code:[30,34,35],codebas:25,collect:[23,30,34],color_axi:34,column:25,combin:[25,31],come:[25,30],comfort:34,commun:37,complet:30,complex:23,compon:[27,30,31,32,33],compos:31,comput:[16,19,22,25,30],concat:32,conduct:[30,31],config:[3,11,30],consecut:11,consid:[25,30],consider:30,consist:32,consol:30,construct:[25,33],contain:[16,19,22,25,27,30,34],content:[24,25,27,30,35],context:[25,37],contrast:[16,19,22],conv2d_11:32,conv2d_13:32,conv2d_15:32,conv2d_18:32,conv2d_1:32,conv2d_20:32,conv2d_22:32,conv2d_25:32,conv2d_27:32,conv2d_29:32,conv2d_3:32,conv2d_6:32,conv2d_8:32,converg:[34,35,36],convert:[5,6,7,8,25,26,27,31,32,34,35,36,37],convert_image_datauri_to_blob:34,copi:[34,36],correct:[30,32],correctli:30,correspond:11,cos_sim:[16,19,22],cosim:[16,19,22],cosin:[3,16,19,22,25],cosinesiameseloss:[0,2,3,10,11,14,16,17,19,20,22,33],cosinetripletloss:[11,14,16,17,19,20,22,33],cost:37,covid19:35,covidqa:35,cpp:[34,36],cpu:[0,10,14,17,20,34],creat:[11,14,17,20,23],csrc:[34,36],csv:25,cuda:[10,14,17,20],current:[11,14,17,20],d8aaaaaaaaeqaaaaaaaaaxa:25,d:[16,19,22,25,27,34],da1:25,da2:25,dam_path:3,danger:30,data:[1,5,6,7,8,9,10,11,14,17,20,21,23,26,27,30,31,33,37],data_gen:34,dataiter:3,dataset:[0,10,14,17,20,25,34,35,36,37],dc315d50:35,debug:[25,30],deep:[1,25,31,37],def:[32,33,34,35],defailt:[10,14,17,20],defin:[5,6,7,8,32],deliv:37,demo:[25,35,36],denot:33,dens:[5,25,32,33,35,36,37],dense_1:32,dense_2:32,depend:30,design:[30,37],desir:[16,19,22,32],detect:27,determin:30,develop:33,devic:[0,10,14,17,20],dict:[0,1,3,5,6,7,8,10,14,17,20,23],dictionari:23,differ:[11,16,19,22,25,30,31,32],dim:[32,33,35,36,37],dimens:[27,32],dimension:[5,6,7,8,32,35,37],direct:[32,33,35],directli:[25,30,32],discuss:37,displai:[4,5],dist:[16,19,22],dist_neg:[16,19,22],dist_po:[16,19,22],distanc:[16,19,22],dive:37,divers:30,divid:30,dnn:[1,5,6,7,8,27],dnn_model:1,doc:[3,30,34,36],doctor:35,document:[1,9,25,27,30,33,34,35,36],documentarrai:[1,25,27,34,35,36],documentarraylik:[0,1],documentarraymemap:25,documentarraymemmap:[1,25,34,35,36],documentsequ:[1,10,14,17,20],doe:[9,26,34,36,37],domain:37,don:[26,37],done:[30,34,36],download:34,dropout_35:32,dropout_38:32,dtype:[25,35,36],dure:3,e:[9,23,25],each:[9,10,14,17,20,25,27,30,33,34],easi:[26,37],easier:30,ecosystem:37,either:[11,14,17,20,25,32,33],ell_:33,emb:3,embed:[1,5,6,7,8,10,11,14,16,17,19,20,22,25,26,27,30,31,32,33,34,37],embed_model:[2,10,11,14,17,20,26,30,32,33,35,36,37],embedding_1:32,embedding_dim:[32,33,35],embedding_lay:[5,6,7,8],enabl:34,end:[10,14,17,20,32],endpoint:3,engin:37,enpow:1,ensur:[16,19,22],epoch:[0,10,11,14,17,20,30,33],epsilon:[10,14,17,20],equal:[16,19,22],estim:[30,34],euclidean:[16,19,22],euclideansiameseloss:[11,14,16,17,19,20,22,33],euclideantripletloss:[11,14,16,17,19,20,22,33],eval_data:[0,10,11,14,17,20,25,33],evalu:[10,14,17,20,25],event:37,everi:[25,30,35,36,37],everyth:30,exampl:[9,11,27,30,34,35,36,37],except:32,executor0:30,executor1:30,executor:[0,2,30,34,36],exhaust:33,exist:[32,33],expect:[3,9],experi:[30,34,37],express:25,extend:25,extra:3,extrem:[26,37],f4:36,f8:25,f:20,factor:9,fail:3,fals:[0,2,3,4,5,6,7,8,9,10,14,17,20,25,32],far:30,fashion:[9,37],faster:30,fc1:32,fc2:32,featur:37,feed:[3,25,33,35,36,37],feedback:[25,35,36],feel:34,few:30,field:[3,5,6,7,8,30],file:[10,23,30,34],filepath:[14,23],fill:[3,25,27,35],find:[26,30,32,37],fine:[26,32,37],finetun:[25,26,27,30,31,32,33],first:[16,19,22,32,33,34,35,36],fit:[0,2,3,10,11,14,17,20,25,27,32,34,35,36,37],fix:[9,25,30],flatten:[32,33,36,37],flatten_1:32,flatten_input:32,float32:[0,4,5,6,7,8,32],flow:[30,34,36],fly:[25,26,37],folder:10,follow:[25,26,30,32,33,34,35,36,37],form:33,format:[33,34,35,36],forward:[19,22,32,33,35],forwardref:0,found:33,four:25,frac:33,framework:[1,10,32,34,35,36,37],freeli:33,freez:[0,4,5,6,7,8,32,34,37],from:[3,5,6,7,8,11,14,17,20,23,25,30,32,33,34,35,36,37],from_fil:34,frontend:[25,30,37],frozen:32,ftexecutor:3,full:34,fundament:32,further:30,g:[9,23],gatewai:30,gener:[5,6,7,8,9,25,27,30,32,33,34,35,36],general_model:[26,30,37],generate_fashion_match:[9,25,33,36,37],generate_qa_match:[9,25,33,35],get:[5,9,11,30,32,34,35,36,37],get_dataset:13,get_embed_model:3,get_framework:1,give:[1,9,32],given:[27,30,32,33,34,36],go:[35,36],good:25,goodby:25,got:[26,37],gpu:[10,14,17,20,34],grai:9,graph:25,grayscal:[9,25],grid:30,groundtruth:25,h236cf4:25,h:9,ha:[25,30,32],hallo:25,hand:37,handl:30,hanxiao:[30,34,36],have:[9,26,30,32,34,35,37],hello:[25,35,36],help:[26,34,37],helper:[0,10,11,12,24,30,32,34],henc:25,here:[16,19,22,25,32,33,34],high:[25,32],hire:37,hit:30,hopefulli:[34,35,36],how:[25,30,32,34,37],howev:[25,33],http:[30,34,36],httpruntim:30,huggingfac:32,human:[27,30,37],i8:[25,35],i:[25,30,33,34,36],ical:37,id:[25,35,36],idea:[1,37],ident:[5,6,7,8,37],identityceleba:34,ignor:[30,34],illustr:25,imag:[9,25,27,30,33,34],imaga:9,imagenet:34,img_align_celeba:34,implement:[1,30,32,33,34],improv:[25,34,35,36,37],in_featur:[32,33,35,36,37],includ:[5,6,7,8,23,25],include_identity_lay:[5,6,7,8],index:[30,37],indic:30,info:[5,6,7,8],inform:[1,25,32],initi:[23,33],inject:3,input:[1,5,6,7,8,11,14,15,16,17,18,19,20,21,22,25,27,30,32,33],input_dim:[32,33,35],input_dtyp:[0,4,5,6,7,8,32],input_s:[0,4,5,6,7,8,32,34],input_shap:[32,33,36,37],inspect:34,inspir:[35,36],instal:37,instanc:[11,14,17,20,25,30],int64:32,integ:[1,5,6,7,8,9],integr:37,interact:[0,25,26,31,37],intern:[34,36],introduc:25,intuit:37,invert:30,io:18,is_seq_int:1,is_sim:[16,19,22],is_testset:[9,33],isspeci:30,item:25,iter:[1,32],its:[25,32,33,34,36],ivborw0k:36,j:33,jina:[1,3,25,30,34,35,36,37],jpg:34,json:23,just:30,k:[30,34,35,36],keep:30,kei:[1,14,17,20,23,30,33],kera:[0,1,4,10,32,33,34,35,36,37],kerastailor:6,kerastun:14,keyboard:[30,34,35,36],keyword:[3,10,14,17,20],known:[16,19,22],kwarg:[2,3,4,5,10,11,14,16,17,19,20,22,33],label:[0,9,11,24,26,27,31,33,37],labeled_dam_path:3,labeled_data:[26,30,37],labler:34,lambda:33,languag:25,larger:30,last:[5,6,7,8,32,34,37],lastcel:[32,33,35],lastcell_3:32,latest:37,layer:[1,5,6,7,8,16,32,33,34,35,36,37],layer_nam:[0,4,5,6,7,8,32],layerinfotyp:1,learn:[10,14,17,20,25,26,30,31,37],learning_r:[0,10,14,17,20],least:25,left:[30,33,35],length:[9,25],let:[32,33,34,35,36,37],level:[25,32,37],leverag:34,licens:37,like:[25,27,30],linear:[32,33,34,35,36,37],linear_2:32,linear_33:32,linear_34:32,linear_36:32,linear_39:32,linear_4:32,linear_5:32,liner:37,linux:37,list:[1,5,6,7,8,16,19,22,23,30,32],live:37,load:[30,32],loader:1,local:[30,34,36],localhost:[30,34,36],lock:30,logic:25,look:[30,32,35,36],loop:37,loss:[0,2,3,10,11,14,17,20,23,34,36],lstm_2:32,luckili:[26,37],m1:25,m2:25,m3:25,m:25,machin:34,maco:37,made:25,mai:[30,32,34,36],main:30,make:[33,34,36,37],mani:30,manual:[25,30,32],map:3,margin:[16,19,22],match:[9,27,30,33],mathbf:33,max:[16,19,22,33],max_seq_len:9,maximum:[9,30],maxpool2d_10:32,maxpool2d_17:32,maxpool2d_24:32,maxpool2d_31:32,maxpool2d_5:32,mean:[25,30,34,35,36],meaning:30,meanwhil:[26,37],meet:37,member:37,memmap:1,meta:3,method:[10,14],metric:[3,23],micro:32,mile:37,mime_typ:25,minimum:33,mix:30,mlp:[35,37],mnist:[9,36,37],model:[0,1,4,5,6,7,8,10,11,14,17,20,25,26,27,30,31,33,37],model_path:10,modul:[24,32,33,35,37],momentum:[10,14,17,20],mond:25,month:37,more:[25,26,30,37],most:30,mous:[34,35,36],move:[10,14,17,20],multi:30,multipl:11,must:[9,33],n:[16,19,22,33],name:[1,5,6,7,8,11,14,17,20,23,32],nativ:23,nb_param:32,ndarrai:[9,25,27],nearest:[25,30],need:[7,14,17,20,25,26,30,32,33,34,37],neg:[9,16,19,22,25,30,33],neg_valu:9,neighbour:[25,30],nesterov:[10,14,17,20],network:[1,11,16,19,22,25,30,31,34,36,37],neural:[1,25,31,37],newli:30,next:[3,27],nn:[19,22,32,33,35,36,37],non:[32,34,36],none:[0,2,3,4,5,6,7,8,9,10,11,14,17,20,23,32],note:[5,6,7,8,11,25,30,34,35,36],now:[25,26,32,34,35,36,37],np:25,num_embed:[32,33,35],num_neg:[9,33],num_po:[9,33],num_tot:9,number:[9,10,14,16,17,19,20,22,23,25,30],numer:23,numerictyp:23,numpi:[23,25,34,36],object:[1,11,12,14,16,17,19,20,22,23,25,27,35,36],observ:[30,32],off:32,often:[25,32],onc:30,one:[11,25,33,37],onli:[25,30,33,34],open:[30,34],opensourc:37,oper:[30,32],optim:[0,1,10,14,16,17,19,20,22,33],optimizer_kwarg:[0,10,14,17,20],option:[0,5,6,7,8,9,10,11,14,17,20,23,32],organ:25,origin:[5,6,7,8,25,32,34],other:[25,33,37],otheriws:[16,19,22],otherwis:25,our:[34,35,36,37],out:[26,32,33,35,37],out_featur:[32,33,35,36,37],output:[5,6,7,8,26,27,30,32,37],output_dim:[0,4,5,6,7,8,26,30,32,33,34,35,37],output_shape_displai:32,over:[16,19,22,32],own:[25,32],p:33,packag:24,pad:35,paddl:[0,1,4,10,32,33,34,35,36,37],paddlepaddl:37,paddletailor:[5,6,7,8],paddletun:17,pair:[16,19,22,33],param:3,paramet:[1,3,5,6,7,8,9,10,11,14,16,17,19,20,22,23,32,33],parent:25,part:[5,6,7,8,30],particular:25,pass:[3,5,10,14,17,20],path:[10,14,17,20,30,34],per:[9,25],perceptron:32,perfect:[26,37],perform:[30,32,37],pip:37,pipelin:[35,36],place:11,plain:25,pleas:34,png:36,pool:30,port_expos:[0,2],pos_valu:9,posit:[9,16,19,22,25,30,33],positv:30,potenti:[5,6,7,8],power:37,pre:34,precis:[35,36],predict:[27,32,33],prepar:25,preserv:32,pretrain:[33,37],previou:32,primit:25,print:25,privat:[30,34,36],problem:[32,33],procedur:[30,34,35,36],process:25,produc:[11,14,17,20,34,35,36],product:[30,37],program:[34,36],project:31,promis:37,properti:[5,11],propos:30,protect:[34,36],protocol:[30,34,36],provid:[25,26,32,37],prune:37,purpos:[25,30],py:[30,34,36],python:[23,33,37],pytorch:[0,4,9,10,32,33,34,35,36,37],pytorchtailor:[5,6,7,8],pytorchtun:20,qa:[9,35],qualiti:[35,36],queri:[30,37],question:[25,26,27,37],quickli:[26,37],rais:[30,34,36],randomli:25,rate:[10,14,17,20],ratio:30,re:37,readi:[30,34,36],real:9,recommend:[30,35,36],record:23,redoc:[30,34,36],reduc:37,reduct:37,refer:33,reflect:25,regress:27,reject:[34,35,36],relat:[25,30,33],relev:[25,30],reli:25,relu:[32,33,36,37],relu_12:32,relu_14:32,relu_16:32,relu_19:32,relu_21:32,relu_23:32,relu_26:32,relu_28:32,relu_2:32,relu_30:32,relu_34:32,relu_37:32,relu_3:32,relu_4:32,relu_7:32,relu_9:32,remain:[30,32],rememb:[35,36],remov:[5,6,7,8,32],render:[30,35],replac:32,repres:[16,19,22,25,30,33],request:3,requir:[5,6,7,8,32,34],rescal:9,reset:33,resnet50:34,respect:[33,34],rest:[34,36],restrict:27,result:[30,34,35,36],rgb:9,rho:[10,14,17,20],rich:37,right:[30,33],rmsprop:[10,14,17,20],round:[34,35,36],row:25,run:34,runtim:[3,30],runtime_arg:3,runtime_backend:[0,2],runtimebackend:30,runtimeerror:30,s:[10,25,27,30,32,34,35,36,37],sai:32,same:[16,19,22,25,27,30,35,36],sampl:[25,30],save:[3,10,11,14,17,20,23,30],save_path:33,scalar:23,scalarsummari:23,scale:9,scenario:31,score:25,scratch:32,script:34,search:[25,35,37],second:[16,19,22,30,34,37],section:30,see:[5,6,7,8,30,32,34,36,37],select:[30,32,35],self:[32,33,35],semant:35,sentenc:25,sequenc:[1,5,6,7,8,9,27],sequenti:[32,33,35,36,37],session:30,set:[5,6,7,8,9,30,32,35,36,37],set_wakeup_fd:30,setup:30,sever:[34,35,36],sgd:[10,14,17,20],shape:[5,6,7,8,25,27,35,36],share:25,shortcut:30,shot:[25,30],should:[9,16,19,22,25,30,34],show:30,siames:[11,16,19,22,33,37],siamesedataset:[15,18,21],siamesemixin:[12,15,18,21],side:30,signal:30,similar:[16,19,22,34,35,36],simpli:[25,26,32,37],simul:9,singl:25,size:[5,6,7,8,10,14,17,20,27,30,34],skip_identity_lay:[6,7,8],slack:37,slower:30,smaller:[30,34],smooth:37,so:[25,30,34,35,36],solid:1,solut:37,solv:[32,33],some:[16,19,22,25,26,30,32,37],sourc:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],space:30,spawn:34,specif:[10,30,32,37],specifi:[30,33,34],spinner:30,stabil:[16,19,22],start:[25,30,34],stat:30,statist:30,step:[32,33,34,36],still:30,store:[23,25,30,35],store_data:3,str:[0,1,5,6,7,8,10,11,14,17,20,23,32],stream:37,stronli:[35,36],submiss:30,submit:30,submodul:24,subpackag:24,subscrib:37,success:30,summar:33,summari:[0,5,6,7,8,10,25,32],summarycollect:[0,10,11,14,17,20,23],support:[10,14,17,20,34,36],suppos:30,supposedli:[34,36],suppress:[34,36],sure:[33,34,37],swagger:[30,34,36],synthet:[9,25],system:30,t:[26,34,37],tabl:[5,32],tag:[25,27,30,35,36],tailor:[0,1,24,26,27,31,34,37],take:34,take_batch:3,taken:25,talk:37,target:[16,19,22],task:[25,35,36,37],tell:[1,30,35],tensor:[5,6,7,8,16,19,22,34,36],tensor_numpi:[34,36],tensorflow:[16,32,33,34,35,36,37],termin:[30,34],test:9,text:[9,25,27,30,35],textbox:30,tf:[32,33,34,35,36,37],than:25,thei:[16,19,22,32],them:[25,32,36],thi:[1,7,11,16,19,22,25,26,30,32,33,34,35,36,37],thread:[0,2,30,34],three:[16,19,22,31,33],through:[35,36],time:[11,25,30,32],to_dataturi:34,to_embedding_model:[0,4,5,6,7,8,26,30,34,37],to_ndarrai:9,togeth:30,together:[10,14,17,20],token:[9,25,27],tool:37,top:[30,34,35,36],topk:30,torch:[1,20,21,22,32,33,34,35,36,37],torchvis:[32,34],total:9,toydata:[0,24,25,33,35,36,37],tp:1,train:[1,10,14,16,17,19,20,22,25,26,27,30,33,34,35,36,37],train_data:[0,2,10,11,14,17,20,25,26,30,33,34,35,36,37],trainabl:32,trainer:[34,36],transform:33,trigger:[34,36],triplet:[11,16,19,22,33,37],tripletdataset:[15,18,21],tripletmixin:[12,15,18,21],tuesdai:37,tune:[25,26,31,32,34,35,36,37],tuner:[0,24,26,27,30,31,32,34,37],tupl:[0,5,6,7,8,16,19,22,32],tutori:[35,36,37],two:[19,22,25,30,32],txt:34,type:[0,1,2,4,5,6,7,8,9,10,11,14,17,19,20,22,23,25,30,32,34,35,36],typevar:1,ui:[30,31,34,35,36],under:[31,37],underli:[34,36],underneath:34,union:[1,10,11,14,17,20,23],unknown:[30,34],unlabel:[27,30],unlabeled_data:[26,30,37],unlock:37,unrel:[30,33],up:[30,34,35,36,37],upsampl:9,uri:36,url:30,us:[1,5,6,7,8,10,14,16,17,19,20,22,25,26,31,32,33,34,35,36],usag:[26,37],user:[34,36,37],userwarn:[30,34,36],util:[21,34,36],valid:25,valu:[3,9,10,14,16,17,19,20,22,23,25,30],valueerror:1,ve:[26,37],vector:[33,35,36,37],veri:30,version:34,via:[26,31,32,33,37],video:37,view:35,vision:[32,34],visual:[34,36],w:[9,30],wa:34,wai:[25,30,33,37],wait:30,want:[32,34,35,36,37],warn:[34,36],we:[9,25,32,33,34,35,36,37],web:25,wedg:[16,19,22],weight:[5,6,7,8,11,30,32,33,34,37],welt:25,what:[26,37],whatev:34,when:[3,5,6,7,8,16,19,22,25,30,33,34,37],where:[10,14,16,17,19,20,22,23,25,27,33,37],wherea:[25,30,33],whether:[25,32],which:[10,14,17,20,25,26,30,32,34,35,37],wiedersehen:25,without:1,work:[30,32,33,34,37],world:25,worri:[26,33,37],wrap:33,write:[33,34,35,36,37],writeabl:[34,36],written:[32,33],wrong_answ:[25,35],x:[9,27,32,33,35],y_:33,yaml:3,ye:[25,26,37],yet:[30,37],yield:[34,35,36],you:[5,6,7,8,14,17,20,25,26,30,32,33,34,35,36,37],your:[25,26,30,32,33,34,35,37],youtub:37,zip:34,zoo:32,zoom:37},titles:["finetuner package","finetuner.helper module","finetuner.labeler package","finetuner.labeler.executor module","finetuner.tailor package","finetuner.tailor.base module","finetuner.tailor.keras package","finetuner.tailor.paddle package","finetuner.tailor.pytorch package","finetuner.toydata module","finetuner.tuner package","finetuner.tuner.base module","finetuner.tuner.dataset package","finetuner.tuner.dataset.helper module","finetuner.tuner.keras package","finetuner.tuner.keras.datasets module","finetuner.tuner.keras.losses module","finetuner.tuner.paddle package","finetuner.tuner.paddle.datasets module","finetuner.tuner.paddle.losses module","finetuner.tuner.pytorch package","finetuner.tuner.pytorch.datasets module","finetuner.tuner.pytorch.losses module","finetuner.tuner.summary module","finetuner","Data Format","One-liner <code class=\"docutils literal notranslate\"><span class=\"pre\">fit()</span></code>","Glossary","&lt;no title&gt;","&lt;no title&gt;","Labeler","Overview","Tailor","Tuner","Finetuning Pretrained ResNet for Celebrity Face Search","Finetuning Bi-LSTM for Question-Answering","Finetuning MLP for Fashion Image Search","Welcome to Finetuner!"],titleterms:{"1":25,Is:25,One:26,advanc:30,all:25,answer:35,argument:33,base:[5,11],bi:[32,35],bidirect:33,build:[35,36],celeba:34,celebr:34,content:[0,2,4,6,7,8,10,12,14,17,20],control:30,covid:[25,33],data:[25,34,35,36],dataset:[12,13,15,18,21],displai:32,embed:[35,36],exampl:[25,32,33],executor:3,face:34,fashion:[25,33,36],field:25,finetun:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,34,35,36,37],fit:[26,30,33],format:25,glossari:27,have:25,helper:[1,13],imag:36,interact:[30,34,35,36],interfac:30,join:37,kera:[6,14,15,16],label:[2,3,25,30,34,35,36],liner:26,load:34,loss:[16,19,22,33],lstm:[32,33,35],match:25,method:[30,32,33],mlp:[32,33,36],mnist:[25,33],model:[32,34,35,36],modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],next:37,okai:25,overview:31,packag:[0,2,4,6,7,8,10,12,14,17,20],paddl:[7,17,18,19],panel:30,prepar:[34,35,36],pretrain:[32,34],progress:30,put:[34,35,36],pytorch:[8,20,21,22],qa:[25,33],question:[30,35],quick:37,requir:25,resnet:34,run:30,save:33,search:[34,36],simpl:[32,33],sourc:25,start:37,step:37,submodul:[0,2,4,10,12,14,17,20],subpackag:[0,4,10],summari:23,supervis:25,support:37,tailor:[4,5,6,7,8,30,32],tip:32,to_embedding_model:32,togeth:[34,35,36],toydata:9,tune:33,tuner:[10,11,12,13,14,15,16,17,18,19,20,21,22,23,33],understand:25,us:[30,37],usag:31,user:30,vgg16:32,view:30,welcom:37,without:30}})