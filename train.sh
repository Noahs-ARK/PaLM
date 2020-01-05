curr_dir="$PWD"
mkdir -p model
mkdir -p log
gpu_id=2
export CUDA_VISIBLE_DEVICES=${gpu_id}

bptt=70
batch_size=20
model="SPAN_LSTM"
finetuning=400
nonmono=15
chart_layer=3
nlayers=3
residual_gate=True
rb=False
num_cpu=1

optimizer="sgd"
lr=30
clip=0.25

dropout=0.4
dropouti=0.4
dropoute=0.1
dropouth=0.3
dropoutho=0.1
wdrop=0.5
max_span_length=10
gate=True
aff=True
activation="relu"
use_pos=False
len_feat=True
fw=True
bw=True
emsize=400
cxtsize=400
rrnn_size=400
parser_size=100
nhid=1120
gamma=0.01

prefix=fw${fw}.bw${bw}.cxt${cxtsize}.rrnn${rrnn_size}.ps${parser_size}.gamma${gamma}.act${activation}.rb${rb}.msl${max_span_length}.lr${lr}.nhid${nhid}.dp${dropout}.dpi${dropouti}.dpe${dropoute}.dph${dropouth}.dropoutho${dropoutho}.wd${wdrop}.lf${len_feat}.g${gate}.aff${aff}.pos${use_pos}
save=model/${prefix}

nohup ~/bin/python3.6 main.py \
--optimizer ${optimizer} \
--batch_size ${batch_size} \
--finetuning ${finetuning} \
--cxtsize ${cxtsize} \
--emsize ${emsize} \
--rrnn_size ${rrnn_size} \
--parser_size ${parser_size} \
--gamma ${gamma} \
--data data/penn \
--data_ptb data/ptb/MRG \
--dropout ${dropout} \
--dropouti ${dropouti} \
--dropoute ${dropoute} \
--dropouth ${dropouth} \
--dropoutho ${dropoutho} \
--wdrop ${wdrop} \
--seed 31415 \
--model ${model} \
--save ${save} \
--nonmono ${nonmono} \
--cuda True \
--bptt ${bptt} \
--lr ${lr} \
--nhid ${nhid} \
--clip ${clip} \
--max_span_length ${max_span_length} \
--nlayers ${nlayers} \
--chart_layer ${chart_layer} \
--gate ${gate} \
--aff ${aff} \
--residual_gate ${residual_gate} \
--activation ${activation} \
--len_feat ${len_feat} \
--use_pos ${use_pos} \
--right_branching ${rb} \
--fw_rrnn ${fw} \
--bw_rrnn ${bw} \
> log/${prefix} &
