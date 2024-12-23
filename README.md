# Opinion Tree Parsing for Aspect-based Sentiment Analysis
Code and data for Opinion Tree Parsing for Aspect-based Sentiment Analysis(Findings of ACL2023)

Xiaoyi Bao, Xiaotong Jiang, Zhongqing Wang, Yue Zhang, and Guodong Zhou. 2023. Opinion Tree Parsing for Aspect-based Sentiment Analysis. In Findings of the Association for Computational Linguistics: ACL 2023, pages 7971–7984, Toronto, Canada. Association for Computational Linguistics.

## Environment
    conda create -n OTP python=3.7.0
    conda activate OTP
    conda install -c conda-forge spacy

## Requirement (Use pip install)
    benepar
    transformers=4.23.1 
    torch==1.10.0
    numpy==1.18.1 
    pytokenizations
    tqdm
## Data preprocessing 

    python ./data/absa/process_data.py
    

## Train for Laptop domain

    python src/main.py train --use-pretrained --model-path-base ./model --batch-size 128 --pretrained-mode t5-base


## Train for Restaurant domain 

Modify the Line 711/713 in src/main.py to corresponding pathes to Restaurant files. 
    
    
## Inference once finished training

    python src/main.py test --model-path lap_t5_dev=0.39.pt  --test-path data/absa/lap_test.txt
    
## Cite
    @inproceedings{bao-etal-2023-opinion,
        title = "Opinion Tree Parsing for Aspect-based Sentiment Analysis",
        author = "Bao, Xiaoyi  and
          Jiang, Xiaotong  and
          Wang, Zhongqing  and
          Zhang, Yue  and
          Zhou, Guodong",
        booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.findings-acl.505",
        doi = "10.18653/v1/2023.findings-acl.505",
        pages = "7971--7984",
        abstract = "Extracting sentiment elements using pre-trained generative models has recently led to large improvements in aspect-based sentiment analysis benchmarks. These models avoid explicit modeling of structure between sentiment elements, which are succinct yet lack desirable properties such as structure well-formedness guarantees or built-in elements alignments. In this study, we propose an opinion tree parsing model, aiming to parse all the sentiment elements from an opinion tree, which can explicitly reveal a more comprehensive and complete aspect-level sentiment structure. In particular, we first introduce a novel context-free opinion grammar to normalize the sentiment structure. We then employ a neural chart-based opinion tree parser to fully explore the correlations among sentiment elements and parse them in the opinion tree form. Extensive experiments show the superiority of our proposed model and the capacity of the opinion tree parser with the proposed context-free opinion grammar. More importantly, our model is much faster than previous models.",
    }
