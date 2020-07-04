#export PROJECT=ia376-1s2020-ptt5
export PROJECT=ia376-1s2020-ptt5-2-282301
export ZONE=europe-west4-a
export BUCKET=gs://ptt5-1/
export TPU_NAME="$HOSTNAME"
# export DATA_DIR="${BUCKET}/your_data_dir"
# export MODEL_DIR="${BUCKET}/your_model_dir"

gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE

# Start TPU
alias tpu-start="gcloud compute tpus start $TPU_NAME"

# Stop TPU
alias tpu-stop="gcloud compute tpus stop $TPU_NAME"
