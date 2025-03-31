model_name=Olympus
MODELDIR=ckpts/$model_name

python predict.py \
  --prompt "Generate an image of a fluffy orange cat lounging on a windowsill, with sunlight streaming through the glass and casting soft shadows to create a cozy atmosphere. Next, change the cat’s color to white, in the following step, produce a high-resolution 3D model based on the modified image." \
  --model-path $MODELDIR \
  --temperature 0 \
  --conv-mode v0
