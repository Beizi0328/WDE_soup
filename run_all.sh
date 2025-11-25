cd ~/autodl-tmp/model-soups

python main.py \
  --model-location "$(pwd)/ssd/checkpoints/soup" \
 --cache-predictions --eval-individual-models \
  --data-location data/ \
  --ensemble \
 --opt-ensemble \


    
    
  
  
