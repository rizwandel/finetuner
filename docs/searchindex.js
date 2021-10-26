Search.setIndex({docnames:["api/finetuner","api/finetuner.embedding","api/finetuner.helper","api/finetuner.labeler","api/finetuner.labeler.executor","api/finetuner.tailor","api/finetuner.tailor.base","api/finetuner.tailor.keras","api/finetuner.tailor.paddle","api/finetuner.tailor.pytorch","api/finetuner.toydata","api/finetuner.tuner","api/finetuner.tuner.base","api/finetuner.tuner.dataset","api/finetuner.tuner.dataset.helper","api/finetuner.tuner.keras","api/finetuner.tuner.keras.datasets","api/finetuner.tuner.keras.losses","api/finetuner.tuner.paddle","api/finetuner.tuner.paddle.datasets","api/finetuner.tuner.paddle.losses","api/finetuner.tuner.pytorch","api/finetuner.tuner.pytorch.datasets","api/finetuner.tuner.pytorch.losses","api/finetuner.tuner.summary","api/modules","basics/data-format","basics/fit","basics/glossary","basics/index","components/index","components/labeler","components/overview","components/tailor","components/tuner","get-started/celeba","get-started/covid-qa","get-started/fashion-mnist","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/finetuner.rst","api/finetuner.embedding.rst","api/finetuner.helper.rst","api/finetuner.labeler.rst","api/finetuner.labeler.executor.rst","api/finetuner.tailor.rst","api/finetuner.tailor.base.rst","api/finetuner.tailor.keras.rst","api/finetuner.tailor.paddle.rst","api/finetuner.tailor.pytorch.rst","api/finetuner.toydata.rst","api/finetuner.tuner.rst","api/finetuner.tuner.base.rst","api/finetuner.tuner.dataset.rst","api/finetuner.tuner.dataset.helper.rst","api/finetuner.tuner.keras.rst","api/finetuner.tuner.keras.datasets.rst","api/finetuner.tuner.keras.losses.rst","api/finetuner.tuner.paddle.rst","api/finetuner.tuner.paddle.datasets.rst","api/finetuner.tuner.paddle.losses.rst","api/finetuner.tuner.pytorch.rst","api/finetuner.tuner.pytorch.datasets.rst","api/finetuner.tuner.pytorch.losses.rst","api/finetuner.tuner.summary.rst","api/modules.rst","basics/data-format.md","basics/fit.md","basics/glossary.md","basics/index.md","components/index.md","components/labeler.md","components/overview.md","components/tailor.md","components/tuner.md","get-started/celeba.md","get-started/covid-qa.md","get-started/fashion-mnist.md","index.md"],objects:{"":{finetuner:[0,0,0,"-"]},"finetuner.embedding":{set_embeddings:[1,1,1,""]},"finetuner.helper":{AnyDNN:[2,2,1,""],AnyDataLoader:[2,2,1,""],AnyOptimizer:[2,2,1,""],DocumentArrayLike:[2,2,1,""],DocumentSequence:[2,2,1,""],LayerInfoType:[2,2,1,""],get_framework:[2,1,1,""],is_seq_int:[2,1,1,""]},"finetuner.labeler":{executor:[4,0,0,"-"],fit:[3,1,1,""]},"finetuner.labeler.executor":{DataIterator:[4,3,1,""],FTExecutor:[4,3,1,""]},"finetuner.labeler.executor.DataIterator":{add_fit_data:[4,4,1,""],requests:[4,5,1,""],store_data:[4,4,1,""],take_batch:[4,4,1,""]},"finetuner.labeler.executor.FTExecutor":{embed:[4,4,1,""],fit:[4,4,1,""],get_embed_model:[4,4,1,""],requests:[4,5,1,""],save:[4,4,1,""]},"finetuner.tailor":{base:[6,0,0,"-"],display:[5,1,1,""],keras:[7,0,0,"-"],paddle:[8,0,0,"-"],pytorch:[9,0,0,"-"],to_embedding_model:[5,1,1,""]},"finetuner.tailor.base":{BaseTailor:[6,3,1,""]},"finetuner.tailor.base.BaseTailor":{display:[6,4,1,""],embedding_layers:[6,6,1,""],summary:[6,4,1,""],to_embedding_model:[6,4,1,""]},"finetuner.tailor.keras":{KerasTailor:[7,3,1,""]},"finetuner.tailor.keras.KerasTailor":{summary:[7,4,1,""],to_embedding_model:[7,4,1,""]},"finetuner.tailor.paddle":{PaddleTailor:[8,3,1,""]},"finetuner.tailor.paddle.PaddleTailor":{summary:[8,4,1,""],to_embedding_model:[8,4,1,""]},"finetuner.tailor.pytorch":{PytorchTailor:[9,3,1,""]},"finetuner.tailor.pytorch.PytorchTailor":{summary:[9,4,1,""],to_embedding_model:[9,4,1,""]},"finetuner.toydata":{generate_fashion_match:[10,1,1,""],generate_qa_match:[10,1,1,""]},"finetuner.tuner":{base:[12,0,0,"-"],dataset:[13,0,0,"-"],fit:[11,1,1,""],keras:[15,0,0,"-"],paddle:[18,0,0,"-"],pytorch:[21,0,0,"-"],save:[11,1,1,""],summary:[24,0,0,"-"]},"finetuner.tuner.base":{BaseDataset:[12,3,1,""],BaseLoss:[12,3,1,""],BaseTuner:[12,3,1,""]},"finetuner.tuner.base.BaseLoss":{arity:[12,5,1,""]},"finetuner.tuner.base.BaseTuner":{arity:[12,6,1,""],embed_model:[12,6,1,""],fit:[12,4,1,""],save:[12,4,1,""]},"finetuner.tuner.dataset":{SiameseMixin:[13,3,1,""],TripletMixin:[13,3,1,""],helper:[14,0,0,"-"]},"finetuner.tuner.dataset.helper":{get_dataset:[14,1,1,""]},"finetuner.tuner.keras":{KerasTuner:[15,3,1,""],datasets:[16,0,0,"-"],get_device:[15,1,1,""],losses:[17,0,0,"-"]},"finetuner.tuner.keras.KerasTuner":{fit:[15,4,1,""],save:[15,4,1,""]},"finetuner.tuner.keras.datasets":{SiameseDataset:[16,3,1,""],TripletDataset:[16,3,1,""]},"finetuner.tuner.keras.losses":{CosineSiameseLoss:[17,3,1,""],CosineTripletLoss:[17,3,1,""],EuclideanSiameseLoss:[17,3,1,""],EuclideanTripletLoss:[17,3,1,""]},"finetuner.tuner.keras.losses.CosineSiameseLoss":{arity:[17,5,1,""],call:[17,4,1,""]},"finetuner.tuner.keras.losses.CosineTripletLoss":{arity:[17,5,1,""],call:[17,4,1,""]},"finetuner.tuner.keras.losses.EuclideanSiameseLoss":{arity:[17,5,1,""],call:[17,4,1,""]},"finetuner.tuner.keras.losses.EuclideanTripletLoss":{arity:[17,5,1,""],call:[17,4,1,""]},"finetuner.tuner.paddle":{PaddleTuner:[18,3,1,""],datasets:[19,0,0,"-"],get_device:[18,1,1,""],losses:[20,0,0,"-"]},"finetuner.tuner.paddle.PaddleTuner":{fit:[18,4,1,""],save:[18,4,1,""]},"finetuner.tuner.paddle.datasets":{SiameseDataset:[19,3,1,""],TripletDataset:[19,3,1,""]},"finetuner.tuner.paddle.losses":{CosineSiameseLoss:[20,3,1,""],CosineTripletLoss:[20,3,1,""],EuclideanSiameseLoss:[20,3,1,""],EuclideanTripletLoss:[20,3,1,""]},"finetuner.tuner.paddle.losses.CosineSiameseLoss":{arity:[20,5,1,""],forward:[20,4,1,""]},"finetuner.tuner.paddle.losses.CosineTripletLoss":{arity:[20,5,1,""],forward:[20,4,1,""]},"finetuner.tuner.paddle.losses.EuclideanSiameseLoss":{arity:[20,5,1,""],forward:[20,4,1,""]},"finetuner.tuner.paddle.losses.EuclideanTripletLoss":{arity:[20,5,1,""],forward:[20,4,1,""]},"finetuner.tuner.pytorch":{PytorchTuner:[21,3,1,""],datasets:[22,0,0,"-"],get_device:[21,1,1,""],losses:[23,0,0,"-"]},"finetuner.tuner.pytorch.PytorchTuner":{fit:[21,4,1,""],save:[21,4,1,""]},"finetuner.tuner.pytorch.datasets":{SiameseDataset:[22,3,1,""],TripletDataset:[22,3,1,""]},"finetuner.tuner.pytorch.losses":{CosineSiameseLoss:[23,3,1,""],CosineTripletLoss:[23,3,1,""],EuclideanSiameseLoss:[23,3,1,""],EuclideanTripletLoss:[23,3,1,""]},"finetuner.tuner.pytorch.losses.CosineSiameseLoss":{arity:[23,5,1,""],forward:[23,4,1,""]},"finetuner.tuner.pytorch.losses.CosineTripletLoss":{forward:[23,4,1,""]},"finetuner.tuner.pytorch.losses.EuclideanSiameseLoss":{arity:[23,5,1,""],forward:[23,4,1,""]},"finetuner.tuner.pytorch.losses.EuclideanTripletLoss":{arity:[23,5,1,""],forward:[23,4,1,""]},"finetuner.tuner.summary":{NumericType:[24,2,1,""],ScalarSequence:[24,3,1,""],Summary:[24,3,1,""]},"finetuner.tuner.summary.ScalarSequence":{floats:[24,4,1,""]},"finetuner.tuner.summary.Summary":{dict:[24,4,1,""],plot:[24,4,1,""],save:[24,4,1,""]},finetuner:{embedding:[1,0,0,"-"],fit:[0,1,1,""],helper:[2,0,0,"-"],labeler:[3,0,0,"-"],tailor:[5,0,0,"-"],toydata:[10,0,0,"-"],tuner:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","data","Python data"],"3":["py","class","Python class"],"4":["py","method","Python method"],"5":["py","attribute","Python attribute"],"6":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:function","2":"py:data","3":"py:class","4":"py:method","5":"py:attribute","6":"py:property"},terms:{"0":[0,10,11,15,17,18,20,21,23,26,27,31,33,34,35,37,38],"00":[27,31,35,37],"000":26,"00010502179":26,"001":[0,11,15,18,21],"002585097":26,"01":27,"011804931":26,"02":27,"028714137":26,"03":35,"06":[27,35],"08":[11,15,18,21],"0e7ec5aa":26,"0e7ec7c6":26,"0e7ecd52":26,"0e7ece7":26,"1":[10,17,20,23,27,31,33,34,35,36,37,38],"10":[0,11,12,15,18,21,26,27,34],"100":[10,27,31,33,35,36,38],"1000":[33,35],"100480":33,"102764544":33,"109":[31,35,37],"11":35,"112":33,"1180160":33,"11ec":[26,36,37],"128":[27,33,34,37,38],"12900":33,"132":[31,35,37],"135":[31,35,37],"14":[27,33],"141":37,"147584":33,"16781312":33,"172":[31,35,37],"1792":33,"18":[31,35,37],"180":35,"19":37,"1bab":37,"1bae":36,"1e":[11,15,18,21],"1e008a366d49":[26,36,37],"1f9f":26,"1faa":26,"2":[10,12,17,20,23,26,27,33,34,36,37,38],"20":35,"224":[33,35],"22900":37,"231":[31,35,37],"2359808":33,"25":27,"25088":33,"2508900":33,"256":[0,11,12,15,18,21,33],"28":[33,34,37,38],"28x28":26,"295168":33,"29672":31,"3":[10,12,17,20,23,26,27,33,35,37,38],"31":37,"32":[27,33,34,36,37,38],"320000":[27,33],"33":37,"3300":33,"36928":33,"3gb":35,"4":[26,27,35,38],"4096":33,"409700":33,"4097000":33,"4128":[27,33],"481":[10,26],"49":37,"5":[26,27,34,38],"5000":[27,33,34,36],"512":33,"52621":37,"53":35,"56":[10,33,37],"5716974480":37,"5794172560":36,"590080":33,"6":[26,35],"60":26,"60000":10,"61130":31,"61622":35,"64":[27,33,34,36],"65":37,"66048":33,"6620":35,"66560":[27,33],"67":37,"67432a92":26,"67432cd6":26,"685":[31,35],"7":[33,37,38],"73856":33,"75":35,"784":33,"784x128x32":34,"9":[11,15,18,21,37,38],"94":[31,35,37],"98":27,"99":[11,15,18,21],"999":[11,15,18,21],"9a49":26,"abstract":[4,6,12],"case":38,"class":[4,6,7,8,9,12,13,15,16,17,18,19,20,21,22,23,24,26,27,33,34,36,37],"default":34,"do":[26,27,33,35,38],"final":[17,20,23,33,35,36,37],"float":[0,11,15,18,21,24,26],"function":[2,4,12,15,18,21,24,26,33,34],"import":[2,26,27,31,33,34,35,36,37,38],"int":[0,3,5,6,7,8,9,10,11,12,15,18,21,24,33],"long":27,"new":[31,33,34,38],"public":[31,35,37,38],"return":[0,1,2,3,5,6,7,8,9,10,11,12,15,18,20,21,23,24,27,33,34,36],"switch":[34,38],"true":[0,2,3,10,26,27,31,33,34,35,36,37,38],"while":31,A:[20,23,28,31],But:[26,27,33,38],By:34,For:[10,12,26,28,31,33,35],If:[3,7,8,9,10,24,26,31,35,36],In:[26,27,31,33,34,35,36,37,38],It:[2,6,7,8,9,10,26,31,33,34,36,38],Its:34,No:[27,38],On:35,One:10,The:[2,3,5,6,7,8,9,10,11,15,17,18,20,21,23,24,26,31,32,34,35,36,37,38],Then:26,To:[5,6,7,8,9,26,33,34],_:[27,33,34,36,38],__init__:8,__module__:2,_i:34,_j:34,_n:34,_p:34,a207:37,a46a:26,a5dd3158:26,a5dd3784:26,a5dd3b94:26,a5dd3d74:26,aaaaaaaa:26,aaaaaaaaaaaaaa:37,aaaaaaaaaaaaaaaa:36,aaaaaaaaekaaaaaaaaawqaaaaaaaabpa:26,abc:[6,12],abl:[31,35],about:[27,34,38],abov:[31,35],ac8a:26,accept:[26,34,35,36,37],access:[31,35,37],accompani:38,accord:34,accur:33,accuraci:37,achiev:33,action:33,activ:[31,32,33,34,35,37,38],actual:[2,24],ad:33,adam:[0,11,15,18,21],adaptiveavgpool2d_32:33,add:[26,33],add_fit_data:4,addit:[3,4,33],address:[31,35,37],adjac:26,adjust:31,advanc:26,affect:31,after:[5,6,7,8,9,26,31,33,34,35,36,37],afterward:[31,33],again:33,agnost:38,ai:38,algorithm:31,alia:[2,24],all:[3,5,6,7,8,9,10,17,20,23,24,27,31,33,34,35,38],allow:[12,26,38],alreadi:[27,31,38],also:[17,20,23,27,31,33,34,36],although:34,alwai:[4,26,27,31],an:[2,3,5,6,7,8,9,11,27,28,31,32,33,34,38],anchor:[17,20,23],ani:[2,6,7,8,9,27,28,31,32,33,36,37,38],answer:[26,27,31,38],anydataload:2,anydnn:[0,1,2,3,5,6,7,8,9,11,12,15,18,21,33],anyoptim:2,anyth:26,apach:38,apart:33,api:[26,33,38],app:38,append:26,applic:[33,35,38],aqaaaaaaaaacaaaaaaaaaamaaaaaaaaa:26,ar:[3,4,6,11,12,15,18,21,26,27,31,33,34,35,36,37,38],ara:4,architectur:[5,6,27,33],arg:[6,11,12,15,17,18,20,21,23],argument:[3,4,11,15,18,21,27,31,35],ariti:[12,14,17,20,23],arrai:[2,26,35,36,37],ask:[28,31],async:31,auf:26,auto:4,avail:[5,6,7,8,9,12,15,18,21,31,35,37],averag:[17,20,23],avoid:4,axi:10,b32d:36,b9557788:37,b:[10,26,28],baaaaaaaaaafaaaaaaaaaayaaaaaaaaa:26,back:38,backend:[3,31,35,37,38],bad:[26,31],bar:[31,36],base64:37,base:[0,2,4,5,7,8,9,11,13,15,16,17,18,19,20,21,22,23,24,31,35,36,37],basedataset:[12,16,19,22],baseexecutor:4,baseloss:[12,17,20,23],basetailor:[6,7,8,9],basetun:[12,15,18,21],batch:[6,7,8,9,11,15,17,18,20,21,23,28],batch_first:[27,33,34,36],batch_siz:[0,11,12,15,18,21],becaus:[26,31],been:[31,33,36],befor:[35,36,37],behav:[27,31],being:[17,20,23],belong:[17,20,23],below:[26,27,31,34,38],besid:31,best:35,beta_1:[11,15,18,21],beta_2:[11,15,18,21],better:[12,17,20,23,28,31,32,35,36,37,38],between:[17,20,23,26,34,38],bewar:35,bidirect:[27,33,36],big:34,bigger:10,blob:[26,36,37],block1_conv1:33,block1_conv2:33,block1_pool:33,block2_conv1:33,block2_conv2:33,block2_pool:33,block3_conv1:33,block3_conv2:33,block3_conv3:33,block3_pool:33,block4_conv1:33,block4_conv2:33,block4_conv3:33,block4_pool:33,block5_conv1:33,block5_conv2:33,block5_conv3:33,block5_pool:33,bonjour:26,bool:[0,2,3,5,6,7,8,9,10,33],both:[17,20,23],bottleneck:33,browser:[31,35],buffer:[26,36,37],build:[26,33,34,38],built:[26,34],butteri:38,button:31,c:[6,7,8,9,10],calendar:38,call:[12,17,26,27,33,34],callabl:[2,3,11,15,18,21],can:[1,5,6,7,8,9,10,26,27,31,32,33,34,35,36,37,38],cannot:26,card:31,center:[11,15,18,21],certain:33,chang:[12,27,31,35],channel:[10,26,38],channel_axi:10,chapter:26,chat:38,chatbot:36,check:[5,6,7,8,9],checkout:[36,37],choic:31,choos:35,chop:33,classif:28,clear_labels_on_start:[0,3,4],click:31,co:34,code:[31,35,36],codebas:26,collect:[24,31,35],column:26,combin:[26,32],come:[26,31],comfort:35,commun:38,complet:31,complex:24,compon:[28,31,32,33,34],compos:32,comput:[1,15,17,18,20,21,23,26,31],concat:33,conduct:[24,31,32],config:[4,12,31],consecut:12,consid:[26,31],consider:31,consist:33,consol:31,construct:[26,34],contain:[17,20,23,26,28,31,35],content:[25,26,28,31,36],context:[26,38],contrast:[17,20,23],conv2d_11:33,conv2d_13:33,conv2d_15:33,conv2d_18:33,conv2d_1:33,conv2d_20:33,conv2d_22:33,conv2d_25:33,conv2d_27:33,conv2d_29:33,conv2d_3:33,conv2d_6:33,conv2d_8:33,converg:[35,36,37],convert:[5,6,7,8,9,26,27,28,32,33,35,36,37,38],convert_image_uri_to_blob:35,copi:[35,37],correct:[31,33],correctli:[27,31],correspond:12,cos_sim:[17,20,23],cosim:[17,20,23],cosin:[3,4,11,17,20,23,26],cosinesiameseloss:[0,3,4,11,12,15,17,18,20,21,23,34],cosinetripletloss:[3,11,12,15,17,18,20,21,23,34],cost:38,covid19:36,covidqa:36,cpp:[35,37],cpu:[0,1,11,15,18,21,35],creat:[12,15,18,21,24],csrc:[35,37],csv:26,cuda:[1,11,15,18,21],current:[12,15,18,21],d8aaaaaaaaeqaaaaaaaaaxa:26,d:[17,20,23,26,28,35],da1:26,da2:26,dam_path:4,danger:31,data:[2,3,5,6,7,8,9,10,11,12,15,18,21,22,24,27,28,31,32,34,38],data_gen:35,dataiter:4,dataset:[0,11,15,18,21,26,35,36,37,38],dc315d50:36,debug:[26,31],deep:[2,26,32,38],def:[27,33,34,35,36],defailt:[11,15,18,21],defin:[6,7,8,9,33],deliv:38,demo:[26,36,37],denot:34,dens:[6,26,33,34,36,37,38],dense_1:33,dense_2:33,depend:[27,31],design:[31,38],desir:[17,20,23,33],detect:28,determin:31,develop:34,devic:[0,1,11,15,18,21],dict:[0,2,4,6,7,8,9,11,15,18,21,24],dictionari:24,differ:[12,17,20,23,26,27,31,32,33],dim:[33,34,36,37,38],dimens:[28,33],dimension:[5,6,7,8,9,33,36,38],direct:[33,34,36],directli:[26,31,33],discuss:38,displai:[5,6],dist:[17,20,23],dist_neg:[17,20,23],dist_po:[17,20,23],distanc:[3,11,17,20,23],dive:38,divers:31,divid:31,dnn:[2,5,6,7,8,9,28],dnn_model:2,doc:[1,4,31,35,37],doctor:36,document:[1,2,10,26,28,31,34,35,36,37],documentarrai:[1,2,26,28,35,36,37],documentarraylik:[0,2],documentarraymemap:26,documentarraymemmap:[1,2,26,35,36,37],documentsequ:[2,3,11,15,18,21],doe:[10,27,35,37,38],domain:38,don:[27,38],done:[27,31,35,37],download:35,dropout_35:33,dropout_38:33,dtype:[26,36,37],dure:4,e:[10,24,26],each:[10,11,15,18,21,26,28,31,34,35],easi:[27,38],easier:31,ecosystem:38,eculidean:[3,11,17,20,23],either:[1,3,12,15,18,21,26,33,34],ell_:34,emb:4,embed:[0,2,3,5,6,7,8,9,11,12,15,17,18,20,21,23,25,26,27,28,31,32,33,34,35,38],embed_model:[1,3,11,12,15,18,21,27,31,33,34,36,37,38],embedding_1:[27,33],embedding_dim:[27,33,34,36],embedding_lay:[5,6,7,8,9],enabl:35,end:[11,15,18,21,33],endpoint:4,engin:38,enpow:2,ensur:[17,20,23],epoch:[0,11,12,15,18,21,27,31,34],epsilon:[11,15,18,21],equal:[17,20,23],estim:[27,31,35],eta:27,euclidean:[17,20,23],euclideansiameseloss:[3,11,12,15,17,18,20,21,23,34],euclideantripletloss:[3,11,12,15,17,18,20,21,23,34],eval:27,eval_data:[0,11,12,15,18,21,26,27,34],evalu:[11,15,18,21,26,27],event:38,everi:[26,31,36,37,38],everyth:31,exampl:[10,12,28,31,35,36,37,38],except:33,executor0:31,executor1:31,executor:[0,3,31,35,37],exhaust:34,exist:[33,34],expect:[4,10],experi:[31,35,38],expos:3,express:26,extend:26,extra:[4,24],extrem:[27,38],f4:37,f8:26,f:21,factor:10,fail:4,fals:[0,3,4,5,6,7,8,9,10,11,15,18,21,26,27,33],far:31,fashion:[10,38],faster:31,fc1:33,fc2:33,featur:38,feed:[4,26,34,36,37,38],feedback:[26,36,37],feel:35,few:31,field:[4,5,6,7,8,9,31],file:[11,24,31,35],filepath:[15,24],fill:[1,4,26,28,36],find:[27,31,33,38],fine:[3,27,33,38],finetun:[26,27,28,31,32,33,34],first:[17,20,23,24,33,34,35,36,37],fit:[0,3,4,11,12,15,18,21,26,28,33,35,36,37,38],fix:[10,26,31],flatten:[33,34,37,38],flatten_1:33,flatten_input:33,float32:[0,5,6,7,8,9,33],flow:[31,35,37],fly:[26,27,38],folder:11,follow:[26,27,31,33,34,35,36,37,38],form:34,format:[34,35,36,37],forward:[20,23,27,33,34,36],forwardref:0,found:[27,34],four:26,frac:34,framework:[2,11,27,33,35,36,37,38],freeli:34,freez:[0,5,6,7,8,9,33,35,38],from:[4,5,6,7,8,9,12,15,18,21,26,27,31,33,34,35,36,37,38],from_fil:35,frontend:[26,31,38],frozen:33,ftexecutor:4,full:35,fundament:33,further:31,g:[10,24],gatewai:31,gener:[5,6,7,8,9,10,26,28,31,33,34,35,36,37],general_model:[27,31,38],generate_fashion_match:[10,26,34,37,38],generate_qa_match:[10,26,27,34,36],get:[6,10,12,15,18,21,24,31,33,35,36,37,38],get_dataset:14,get_devic:[15,18,21],get_embed_model:4,get_framework:2,give:[2,10,27,33],given:[24,28,31,33,34,35,37],go:[36,37],good:26,goodby:26,got:[27,38],gpu:[11,15,18,21,35],grai:10,graph:26,grayscal:[10,26],green:27,grid:31,groundtruth:26,h236cf4:26,h:10,ha:[26,31,33],hallo:26,hand:38,handl:31,hanxiao:[31,35,37],have:[10,27,31,33,35,36,38],hello:[26,36,37],help:[27,35,38],helper:[0,11,12,13,25,31,33,35],henc:26,here:[17,20,23,26,27,33,34,35],high:[26,33],hire:38,hit:31,hopefulli:[35,36,37],how:[26,27,31,33,35,38],howev:[26,34],http:[31,35,37],httpruntim:31,huggingfac:33,human:[28,31,38],i8:[26,36],i:[26,31,34,35,37],ical:38,id:[26,36,37],idea:[2,38],ident:[6,7,8,9,38],identityceleba:35,ignor:[31,35],illustr:26,imag:[10,26,28,31,34,35],imaga:10,imagenet:35,img_align_celeba:35,implement:[2,31,33,34,35],improv:[26,35,36,37,38],in_featur:[27,33,34,36,37,38],includ:[6,24,26],include_identity_lay:6,index:[31,38],indic:31,info:6,inform:[2,7,8,9,26,27,33],initi:34,inject:4,inplac:1,input:[2,5,6,7,8,9,12,15,16,17,18,19,20,21,22,23,26,28,31,33,34],input_dim:[33,34,36],input_dtyp:[0,5,6,7,8,9,27,33],input_s:[0,5,6,7,8,9,27,33,35],input_shap:[33,34,37,38],insid:3,inspect:35,inspir:[36,37],instal:[24,38],instanc:[12,15,18,21,26,31],int64:33,integ:[2,6,7,8,9,10],integr:38,interact:[0,3,26,27,32,38],intern:[35,37],interpret:[7,8,9],introduc:26,intuit:38,invert:31,io:19,is_seq_int:2,is_sim:[17,20,23],is_testset:[10,34],isspeci:31,item:26,iter:[2,33],its:[26,33,34,35,37],ivborw0k:37,j:34,jina:[2,4,26,31,35,36,37,38],jpg:35,json:24,just:31,k:[31,35,36,37],keep:31,kei:[2,15,18,21,24,31,34],kera:[0,1,2,5,11,27,33,34,35,36,37,38],kerastailor:7,kerastun:15,keyboard:[31,35,36,37],keyword:[3,4,11,15,18,21],known:[17,20,23],kwarg:[3,4,5,6,11,12,15,17,18,20,21,23,24,34],label:[0,10,12,25,27,28,32,34,38],labeled_dam_path:4,labeled_data:[27,31,38],labler:35,lambda:[27,34],languag:26,larger:[24,31],last:[5,6,7,8,9,33,35,38],lastcel:[27,33,34,36],lastcell_3:[27,33],latest:38,layer:[2,5,6,7,8,9,17,27,33,34,35,36,37,38],layer_nam:[0,5,6,7,8,9,27,33],layerinfotyp:2,learn:[11,15,18,21,26,27,31,32,38],learning_r:[0,11,15,18,21],least:26,left:[31,34,36],length:[10,26],let:[33,34,35,36,37,38],level:[26,33,38],leverag:35,licens:38,like:[26,28,31],line:35,linear:[27,33,34,35,36,37,38],linear_2:33,linear_33:33,linear_34:33,linear_36:33,linear_39:33,linear_4:[27,33],linear_5:33,liner:38,linspac:24,linux:38,list:[2,5,6,7,8,9,17,20,23,24,31,33],live:38,load:[31,33],loader:2,local:[31,35,37],localhost:[31,35,37],lock:31,logic:26,look:[31,33,36,37],loop:38,loss:[0,3,4,11,12,15,18,21,24,27,35,37],lstm:27,lstm_2:[27,33],luckili:[27,38],m1:26,m2:26,m3:26,m:26,machin:35,maco:38,made:26,mai:[27,31,33,35,37],main:31,make:[34,35,37,38],mani:31,manual:[26,31,33],map:4,margin:[17,20,23],match:[10,28,31,34],mathbf:34,matplotlib:24,max:[17,20,23,34],max_plot_point:24,max_seq_len:[10,27],maximum:[10,24,31],maxpool2d_10:33,maxpool2d_17:33,maxpool2d_24:33,maxpool2d_31:33,maxpool2d_5:33,mean:[26,31,35,36,37],meaning:31,meanwhil:[27,38],meet:38,member:38,memmap:2,meta:4,method:[11,15],metric:[4,24],micro:33,mile:38,mime_typ:26,minimum:34,mix:31,mlp:[36,38],mnist:[10,37,38],model:[0,1,2,3,5,6,7,8,9,11,12,15,18,21,26,28,31,32,34,38],model_path:11,modul:[25,27,33,34,36,38],momentum:[11,15,18,21],mond:26,month:38,more:[26,27,31,38],most:31,mous:[35,36,37],move:[11,15,18,21],multi:31,multipl:12,must:[10,34],n:[17,20,23,34],name:[2,5,6,7,8,9,12,15,18,21,24,27,33],nativ:24,nb_param:[27,33],ndarrai:[10,26,28],nearest:[26,31],need:[8,15,18,21,26,27,31,33,34,35,38],neg:[10,17,20,23,26,31,34],neg_valu:10,neighbour:[26,31],nesterov:[11,15,18,21],network:[2,3,11,12,17,20,23,26,31,32,35,37,38],neural:[2,26,32,38],newli:31,next:[4,28],nn:[20,23,27,33,34,36,37,38],non:[33,35,37],none:[0,1,3,4,5,6,7,8,9,10,11,12,15,18,21,24,33],normalize_image_blob:35,note:[6,7,8,9,12,26,27,31,35,36,37],now:[26,27,33,35,36,37,38],np:26,num_embed:[27,33,34,36],num_neg:[10,27,34],num_po:[10,34],num_tot:10,number:[10,11,15,17,18,20,21,23,24,26,31],numer:24,numerictyp:24,numpi:[24,26,35,37],object:[2,12,13,15,17,18,20,21,23,24,26,27,28,36,37],observ:[31,33],off:33,often:[26,33],onc:31,one:[12,24,26,34,38],onli:[26,27,31,34,35],open:[31,35],opensourc:38,oper:[31,33],optim:[0,2,11,15,17,18,20,21,23,34],optimizer_kwarg:[0,11,15,18,21],option:[0,3,5,6,7,8,9,10,11,12,15,18,21,24,33],organ:26,origin:[5,6,7,8,9,26,33,35],other:[26,34,38],otheriws:[17,20,23],otherwis:26,our:[35,36,37,38],out:[27,33,34,36,38],out_featur:[27,33,34,36,37,38],output:[5,6,7,8,9,24,27,28,31,33,38],output_dim:[0,5,6,7,8,9,27,31,33,34,35,36,38],output_shape_displai:[27,33],over:[17,20,23,27,33],own:[26,33],p:34,packag:25,pad:36,paddl:[0,1,2,5,11,27,33,34,35,36,37,38],paddlepaddl:38,paddletailor:[6,7,8,9],paddletun:18,pair:[17,20,23,34],parallel:3,param:4,paramet:[1,2,3,4,5,6,7,8,9,10,11,12,15,17,18,20,21,23,24,27,33,34],parent:26,part:[6,7,8,9,31],particular:26,pass:[4,6,11,15,18,21,24],path:[11,15,18,21,24,31,35],pea:3,per:[10,26],perceptron:33,perfect:[27,38],perform:[31,33,38],pip:38,pipelin:[36,37],place:12,plain:26,pleas:35,plot:[24,27],png:[27,37],point:24,pool:31,port:3,port_expos:[0,3],pos_valu:10,posit:[10,17,20,23,26,31,34],positv:31,potenti:6,power:38,pre:35,precis:[36,37],predict:[28,33,34],prepar:26,preserv:33,pretrain:[34,38],previou:33,primit:26,print:26,privat:[31,35,37],problem:[33,34],procedur:[27,31,35,36,37],process:[3,26],produc:[7,8,9,12,15,18,21,35,36,37],product:[31,38],program:[35,37],project:32,promis:38,properti:[6,12],propos:31,protect:[35,37],protocol:[31,35,37],provid:[26,27,33,38],prune:38,purpos:[26,31],py:[31,35,37],python:[24,34,38],pytorch:[0,1,5,10,11,27,33,34,35,36,37,38],pytorchtailor:[6,7,8,9],pytorchtun:21,qa:[10,36],qualiti:[36,37],queri:[31,38],question:[26,27,28,38],quickli:[27,38],rais:[31,35,37],randomli:26,rate:[11,15,18,21],ratio:31,re:38,readi:[31,35,37],real:10,recommend:[31,36,37],record:24,redoc:[31,35,37],reduc:38,reduct:38,refer:34,reflect:26,regress:28,reject:[35,36,37],relat:[26,31,34],relev:[26,31],reli:26,relu:[33,34,37,38],relu_12:33,relu_14:33,relu_16:33,relu_19:33,relu_21:33,relu_23:33,relu_26:33,relu_28:33,relu_2:33,relu_30:33,relu_34:33,relu_37:33,relu_3:33,relu_4:33,relu_7:33,relu_9:33,remain:[31,33],rememb:[36,37],remov:[3,5,6,7,8,9,33],render:[27,31,36],replac:33,repres:[17,20,23,26,31,34],request:4,requir:[6,7,8,9,24,27,33,35],rescal:10,reset:34,resnet50:35,respect:[34,35],rest:[35,37],restrict:28,result:[31,35,36,37],rgb:10,rho:[11,15,18,21],rich:38,right:[31,34],rmsprop:[11,15,18,21],round:[35,36,37],row:26,run:35,runtim:[3,4,31],runtime_arg:4,runtime_backend:[0,3],runtimebackend:31,runtimeerror:31,s:[11,26,27,28,31,33,35,36,37,38],sai:33,same:[17,20,23,26,28,31,36,37],sampl:[24,26,31],save:[4,11,12,15,18,21,24,31],save_path:34,scalar:24,scalarsequ:24,scale:10,scenario:32,score:26,scratch:33,script:35,search:[26,36,38],second:[17,20,23,27,31,35,38],section:31,see:[5,6,7,8,9,27,31,33,35,37,38],select:[31,33,36],self:[27,33,34,36],semant:36,sentenc:26,sequenc:[2,6,7,8,9,10,28],sequenti:[27,33,34,36,37,38],session:31,set:[3,5,6,7,8,9,10,31,33,36,37,38],set_embed:1,set_image_blob_channel_axi:35,set_wakeup_fd:31,setup:31,sever:[35,36,37],sgd:[11,15,18,21],shape:[6,7,8,9,26,28,36,37],share:26,shortcut:31,shot:[26,31],should:[10,17,20,23,26,31,35],show:[24,31],siames:[3,11,12,17,20,23,34,38],siamesedataset:[16,19,22],siamesemixin:[13,16,19,22],side:31,signal:31,similar:[17,20,23,35,36,37],simpli:[26,27,33,38],simul:10,singl:26,size:[5,6,7,8,9,11,15,18,21,28,31,35],skip:[7,8,9],skip_identity_lay:[7,8,9],slack:38,slower:31,smaller:[31,35],smooth:38,so:[26,31,35,36,37],solid:2,solut:38,solv:[33,34],some:[17,20,23,26,27,31,33,38],sometim:27,sourc:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],space:31,spawn:35,specif:[11,31,33,38],specifi:[31,34,35],spinner:31,stabil:[17,20,23],start:[26,31,35],stat:31,statist:[27,31],step:[27,33,34,35,37],still:31,store:[7,8,9,24,26,31,36],store_data:4,str:[0,1,2,3,5,6,7,8,9,11,12,15,18,21,24,33],stream:38,stronli:[36,37],submiss:31,submit:31,submodul:25,subpackag:25,subscrib:38,success:31,summar:34,summari:[0,5,6,7,8,9,11,12,15,18,21,26,27,33],support:[3,11,15,18,21,35,37],suppos:31,supposedli:[35,37],suppress:[35,37],sure:[34,35,38],swagger:[31,35,37],synthet:[10,26],system:31,t:[27,35,38],tabl:[5,6,33],tag:[26,28,31,36,37],tailor:[0,2,25,27,28,32,35,38],take:35,take_batch:4,taken:26,talk:38,target:[17,20,23],task:[26,36,37,38],tell:[2,31,36],tensor:[6,7,8,9,17,20,23,35,37],tensor_numpi:[35,37],tensorflow:[15,17,27,33,34,35,36,37,38],termin:[31,35],test:10,text:[10,26,28,31,36],textbox:31,tf:[33,34,35,36,37,38],than:[24,26],thei:[17,20,23,33],them:[26,33,37],thi:[2,8,12,17,20,23,24,26,27,31,33,34,35,36,37,38],thread:[0,3,31,35],three:[17,20,23,32,34],through:[36,37],time:[12,26,27,31,33],to_dataturi:35,to_embedding_model:[0,5,6,7,8,9,27,31,35,38],to_ndarrai:10,togeth:31,together:[11,15,18,21],token:[10,26,28],tool:38,top:[31,35,36,37],topk:31,torch:[2,21,22,23,27,33,34,35,36,37,38],torchvis:[33,35],total:10,toydata:[0,25,26,27,34,36,37,38],tp:2,train:[2,3,11,15,17,18,20,21,23,26,27,28,31,34,35,36,37,38],train_data:[0,3,11,12,15,18,21,26,27,31,34,35,36,37,38],trainabl:[27,33],trainer:[35,37],transform:34,trigger:[35,37],triplet:[3,11,12,17,20,23,34,38],tripletdataset:[16,19,22],tripletmixin:[13,16,19,22],tuesdai:38,tune:[3,26,27,32,33,35,36,37,38],tuned_model:[27,38],tuner:[0,25,27,28,31,32,33,35,38],tupl:[0,5,6,7,8,9,17,20,23,27,33],tutori:[36,37,38],two:[20,23,26,27,31,33],txt:35,type:[0,1,2,3,5,6,7,8,9,10,11,12,15,18,20,21,23,24,26,31,33,35,36,37],typevar:2,ui:[3,24,31,32,35,36,37],under:[32,38],underli:[35,37],underneath:35,union:[1,2,3,11,12,15,18,21,24],unknown:[31,35],unlabel:[28,31],unlabeled_data:[27,31,38],unlock:38,unrel:[31,34],up:[31,35,36,37,38],upsampl:10,uri:37,url:31,us:[1,2,3,5,6,7,8,9,11,15,17,18,20,21,23,26,27,32,33,34,35,36,37],usag:[27,38],user:[35,37,38],userwarn:[31,35,37],util:[22,35,37],valid:26,valu:[4,10,11,15,17,18,20,21,23,24,26,31],valueerror:2,ve:[27,38],vector:[34,36,37,38],veri:31,version:35,via:[27,32,33,34,38],video:38,view:36,vision:[33,35],visual:[24,35,37],w:[10,31],wa:35,wai:[26,31,34,38],wait:31,want:[33,35,36,37,38],warn:[35,37],we:[10,26,33,34,35,36,37,38],web:26,wedg:[17,20,23],weight:[5,6,7,8,9,12,31,33,34,35,38],welt:26,what:[27,38],whatev:35,when:[4,5,6,7,8,9,17,20,23,24,26,31,34,35,38],where:[11,15,17,18,20,21,23,24,26,28,34,38],wherea:[26,27,31,34],whether:[26,33],which:[3,11,15,18,21,26,27,31,33,35,36,38],whose:27,wiedersehen:26,without:2,work:[31,33,34,35,38],world:26,worri:[27,34,38],wrap:34,write:[34,35,36,37,38],writeabl:[35,37],written:[1,33,34],wrong_answ:[26,36],x:[10,27,28,33,34,36],y_:34,yaml:4,ye:[26,27,38],yet:[31,38],yield:[35,36,37],you:[5,6,7,8,9,15,18,21,26,27,31,33,34,35,36,37,38],your:[26,27,31,33,34,35,36,38],youtub:38,zip:35,zoo:33,zoom:38},titles:["finetuner package","finetuner.embedding module","finetuner.helper module","finetuner.labeler package","finetuner.labeler.executor module","finetuner.tailor package","finetuner.tailor.base module","finetuner.tailor.keras package","finetuner.tailor.paddle package","finetuner.tailor.pytorch package","finetuner.toydata module","finetuner.tuner package","finetuner.tuner.base module","finetuner.tuner.dataset package","finetuner.tuner.dataset.helper module","finetuner.tuner.keras package","finetuner.tuner.keras.datasets module","finetuner.tuner.keras.losses module","finetuner.tuner.paddle package","finetuner.tuner.paddle.datasets module","finetuner.tuner.paddle.losses module","finetuner.tuner.pytorch package","finetuner.tuner.pytorch.datasets module","finetuner.tuner.pytorch.losses module","finetuner.tuner.summary module","finetuner","Data Format","One-liner <code class=\"docutils literal notranslate\"><span class=\"pre\">fit()</span></code>","Glossary","&lt;no title&gt;","&lt;no title&gt;","Labeler","Overview","Tailor","Tuner","Finetuning Pretrained ResNet for Celebrity Face Search","Finetuning Bi-LSTM for Question-Answering","Finetuning MLP for Fashion Image Search","Welcome to Finetuner!"],titleterms:{"1":26,Is:26,One:27,advanc:31,all:26,answer:36,argument:34,base:[6,12],bi:[33,36],bidirect:34,build:[36,37],celeba:35,celebr:35,content:[0,3,5,7,8,9,11,13,15,18,21],control:31,covid:[26,34],data:[26,35,36,37],dataset:[13,14,16,19,22],displai:[27,33],embed:[1,36,37],exampl:[26,27,33,34],executor:4,face:35,fashion:[26,34,37],field:26,finetun:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,35,36,37,38],fit:[27,31,34],format:26,glossari:28,have:26,helper:[2,14],imag:37,interact:[31,35,36,37],interfac:31,join:38,kera:[7,15,16,17],label:[3,4,26,31,35,36,37],liner:27,load:35,loss:[17,20,23,34],lstm:[33,34,36],match:26,method:[31,33,34],mlp:[33,34,37],mnist:[26,34],model:[27,33,35,36,37],modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],next:38,okai:26,overview:32,packag:[0,3,5,7,8,9,11,13,15,18,21],paddl:[8,18,19,20],panel:31,prepar:[35,36,37],pretrain:[33,35],progress:31,put:[35,36,37],pytorch:[9,21,22,23],qa:[26,34],question:[31,36],quick:38,requir:26,resnet:35,run:31,save:[27,34],search:[35,37],simpl:[33,34],sourc:26,start:38,step:38,submodul:[0,3,5,11,13,15,18,21],subpackag:[0,5,11],summari:24,supervis:26,support:38,tailor:[5,6,7,8,9,31,33],tip:33,to_embedding_model:33,togeth:[35,36,37],toydata:10,tune:34,tuner:[11,12,13,14,15,16,17,18,19,20,21,22,23,24,34],understand:26,us:[31,38],usag:32,user:31,vgg16:33,view:31,welcom:38,without:31}})