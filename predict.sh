model_name=Olympus
MODELDIR=ckpts/$model_name

python predict.py \
  --prompt "Generate an image of a fluffy orange cat lounging on a windowsill, \
with sunlight streaming through the glass and casting soft shadows to create a cozy atmosphere. Next, \
would it be possible to change the cat's color to white? This change will make it more eye-catching. \
In the following step, produce a high-resolution 3D model based on the modified image. \
At the next point, please show a video of a cat and a dog running on a playground." \
  --model-path $MODELDIR \
  --temperature 0 \
  --conv-mode v0
