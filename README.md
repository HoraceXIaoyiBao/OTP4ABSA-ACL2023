# Opinion Tree Parsing for Aspect-based Sentiment Analysis
Code and data for Opinion Tree Parsing for Aspect-based Sentiment Analysis(ACL2023 Findings)

## Requirement
    benepar
    transformers=4.23.1 
    scipy
    torch=1.10.0
    python==3.7.0
    numpy==1.18.1 
    tqdm
## Data preprocessing 

    python ./data/absa/process_data.py
    

## Train 

    python src/main.py train --use-pretrained --model-path-base ./model --batch-size 128 --pretrained-mode t5-base
    
    
## Inference once finished training

    python src/main.py test --model-path lap_t5_dev=0.39.pt  --test-path data/absa/lap_test.txt
    
    
