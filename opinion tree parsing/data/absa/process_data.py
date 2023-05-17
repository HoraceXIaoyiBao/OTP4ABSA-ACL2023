#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json


# In[4]:


def readfile(file):
    f=open(file,"r")
    
    outputFile=open(file.split(".json")[0]+"_cs_allow_muti_category_dual_missing_front_trans_triple_pair_span.json","w")   
    
    outputFile_only_tree=open("_".join(file.split("_")[:2])+".txt","w")   
    
    sentimentDict={"POS":"Positive","NEU":"Neural","NEG":"Negative"}
    data=[json.loads(i) for i in f.readlines()]
    f.close()
    
    
    general_data=[[j["sentence"],
                   j["aspect"],
                   j["aspectSpan"],
                   j["opinion"],
                   j["opinionSpan"],
                   j["sentiment"], 
                   j["category"]]for j in data]
    
    blanked=0
    nested=0
    cross=0
    diff=0
    left=0

    
    for i in general_data:
        
        
        flag=0
        target=""
        for a in range(len(i[1])):
            for aa in range(a+1,len(i[1])):
                
                if i[1][a]==i[1][aa] and i[6][a]!=i[6][aa] and i[1][a]!="NULL":
                    flag=1
                    target=i[1][a]
        #if flag==1:
           
            #diff+=i[1].count(target)
            #continue
            
        #flag=0
                    
        for a in range(len(i[3])):
            for aa in range(a+1,len(i[3])):
                
                if i[3][a]==i[3][aa] and i[5][a]!=i[5][aa] and i[3][a]!="NULL":
                    flag=1
                    target=i[3][a]
                    
        #if flag==1:
           
            #diff+=i[3].count(target)
            #continue
        
        
        #删除NULL 和 跨
        
        quad_list=[]
        quad_list_single=[]
        quad_list_dual_miss=[]
        
        for a,asp,o,osp,s,c in zip(i[1],i[2],i[3],i[4],i[5],i[6]):
            
            c="cate"
            s="polarity"
            if (a=="NULL" and o=="NULL")  or a=="" or o=="" :
                quad_list_dual_miss.append([ [[a,[-2,-1],c]], [[o,[-1,-1],s]], [-2,-1] ])
                left+=1
                continue
            # elif a=="NULL" :
            #     single+=1
            #     quad_list_single.append([ [[a,asp,c]], [[o,osp,s]], [osp[0],osp[1]] ])
            #     continue
            # elif o=="NULL":
            #     single+=1
            #     quad_list_single.append([ [[a,asp,c]], [[o,osp,s]], [asp[0],asp[1]] ])
            #     continue
                
            elif a==o or a in o or o in a:
                nested+=1
                continue
                
                
                
            if quad_list:
                
                
                # 删除挎
                for q in quad_list:
                    if asp[0] in range(q[2][0],q[2][1]) or \
                       osp[0] in range(q[2][0],q[2][1]) or \
                       asp[1] in range(q[2][0],q[2][1]) or \
                       osp[1] in range(q[2][0],q[2][1]):
                        
                        
                        if (a==q[0][0][0] and asp==q[0][0][1] and len(set([xx[0] for xx in q[0]]))==1) or\
                        (o==q[1][0][0] and osp==q[1][0][1] and len(set([xx[0] for xx in q[1]]))==1):
                            continue
                        else:
                        
                        
                            cross+=1
                            break
                
                
                
                else:  
                    # 一对多
                    for q in quad_list:     
                        if a==q[0][0][0] and asp==q[0][0][1] and asp[0]!=-1 and len(set([xx[0] for xx in q[0]]))==1:
                            if osp[0]==-1:
                                break
                            q[0].append([a,asp,c])
                            q[1].append([o,osp,s])
                            left+=1
                            if osp[0]!=-1:
                                q[2][0]=min(q[2][0],osp[0])
                                q[2][1]=max(q[2][1],osp[1])
                            break

                        elif o==q[1][0][0] and osp==q[1][0][1] and osp[0]!=-1 and len(set([xx[0] for xx in q[1]]))==1:
                            if  asp[0]==-1:
                                break
                            q[0].append([a,asp,c])
                            q[1].append([o,osp,s])
                            left+=1
                            if asp[0]!=-1:
                                q[2][0]=min(q[2][0],asp[0])
                                q[2][1]=max(q[2][1],asp[1])
                            break
                    
                    
                    else:
                        # 单缺
                        if a=="NULL" :
                            left+=1
                            quad_list.append([ [[a,asp,c]], [[o,osp,s]], [osp[0],osp[1]] ])
                            continue
                        elif o=="NULL":
                            left+=1
                            quad_list.append([ [[a,asp,c]], [[o,osp,s]], [asp[0],asp[1]] ])
                            continue 
                        # 一对一
                        else:
                            left+=1
                            quad_list.append([ [[a,asp,c]], [[o,osp,s]], [min(asp[0],osp[0]),max(asp[1],osp[1])] ])
            else:
                if a=="NULL" :
                    left+=1
                    quad_list.append([ [[a,asp,c]], [[o,osp,s]], [osp[0],osp[1]] ])
                    continue
                elif o=="NULL":
                    left+=1
                    quad_list.append([ [[a,asp,c]], [[o,osp,s]], [asp[0],asp[1]] ])
                    continue
                else:
                    left+=1
                    quad_list.append([ [[a,asp,c]], [[o,osp,s]], [min(asp[0],osp[0]),max(asp[1],osp[1])] ])

        # 删除一对多中 的NULL
        for q in quad_list:
            if len(q[0])>1 and len(q[1])>1:
                for ao in range(len(q[0])):
                    if q[0][ao][0]=="NULL" or q[1][ao][0]=="NULL":
                        # print(q)
                        diff+=1
                        idx=ao
                        q[0].pop(ao)
                        q[1].pop(ao)
                        
                        break
                        
                        
                
                
            
            
        if len(quad_list)==0 and len(quad_list_dual_miss)==0:
            continue
            
        if len(quad_list_dual_miss)!=0:
            quad_list.append(quad_list_dual_miss[0])
        
        # 构建tree
        
        tree=["(W null)","(W null)"]+[ "(W " +w+")" for w in i[0].replace("(","-LRB-").replace(")","-RRB-").split() ]
        
        tree_quad_list_sA2mO=[]
        tree_quad_list_sA2sO=[]
        tree_quad_list_mA2sO=[]
        
        tree_quad_list_sA20O=[]
        tree_quad_list_0A2sO=[]
        
        tree_quad_list_dual_missing=[]
        
        opinion_tree_quad_list=[]
        
        for q in quad_list:
            
            
            tree_q=[]
            
            
            
            
            # single aspect to muti opinion
            if len(q[0])>=2 and q[0][0][0]==q[0][1][0]:# <len(q[1]):
                
                
                #################################################################################
                category=list(set(["_".join([ c.upper() for c in q[0][ai][2].split() ]) for ai in range(len(q[0])) ]))
                
                if len(category)==1:
                    category=[category[0]]
                else:
                    category=["_".join([ c.upper() for c in q[0][ai][2].split() ]) for ai in range(len(q[0])) ]
                    
                #if len(q[0].split())>1:
                aspect= [["(A "+w+")" for w in q[0][0][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[0][0][1]]
                
                for c in category:
                    aspect[0][0]="("+c+" "+aspect[0][0]
                    aspect[0][-1]=aspect[0][-1]+")"
                
                opinions=[]
                for oi in range(len(q[1])):
                    
                    sentiment=q[1][oi][2].upper()
                    opinion=[["(O "+w+")" for w in q[1][oi][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[1][oi][1]]
                    
                    
                    
                    # opinion[0][0]="(OPINION "+opinion[0][0]
                    # opinion[0][-1]=opinion[0][-1]+")"
                    
                    opinion[0][0]="("+sentiment+" "+opinion[0][0]
                    
                    opinion[0][-1]=opinion[0][-1]+")"
                    opinions.append(opinion)
                    
                    #################################################################################
                    if len(category)!=1:
                        opinion_tree_quad_list.append([category[oi],q[0][0][0],sentiment,q[1][oi][0],[q[0][0][1][0],q[0][0][1][1]],[q[1][oi][1][0],q[1][oi][1][1]]])
                    else:
                        opinion_tree_quad_list.append([category[0],q[0][0][0],sentiment,q[1][oi][0],[q[0][0][1][0],q[0][0][1][1]],[q[1][oi][1][0],q[1][oi][1][1]]])
                    
                # print(aspect)
                # print(opinions)
                if aspect[1][0]==q[2][0]:
                    aspect[0][0] = "(Q "+aspect[0][0]
                if aspect[1][1]==q[2][1]:
                    aspect[0][-1] = aspect[0][-1]+")"
                    
                    
                for oi in range(len(opinions)):
                    if opinions[oi][1][0]==q[2][0]:
                        opinions[oi][0][0]  = "(Q "+opinions[oi][0][0] 

                    if opinions[oi][1][1]==q[2][1]:
                        opinions[oi][0][-1] = opinions[oi][0][-1]+")"
                    
                    
                tree_q.append(aspect)
                tree_q.append(opinions) 
                tree_quad_list_sA2mO.append(tree_q)
                
                
            # muti aspect to single opinion    
            elif len(q[1])>=2 and  q[1][0][0]==q[1][1][0]: #len(q[1]):
                
                #################################################################################
                #category="_".join([ c.upper() for c in q[0][0][2].split() ])
                sentiment=list(set([q[1][oi][2].upper() for oi in range(len(q[1])) ]))
                if len(sentiment)==1:
                    sentiment=[sentiment[0]]
                else:
                    sentiment=[q[1][oi][2].upper() for oi in range(len(q[1])) ]
                    
                #if len(q[0].split())>1:
                opinion= [["(O "+w+")" for w in q[1][0][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[1][0][1]]
                
                for s in sentiment:
                    opinion[0][0]="("+s+" "+opinion[0][0]
                    opinion[0][-1]=opinion[0][-1]+")"
                
                aspects=[]
                for ai in range(len(q[0])):
                    #print(q[0])
                    
                    
                    category="_".join([ c.upper() for c in q[0][ai][2].split() ])
                    aspect=[["(A "+w+")" for w in q[0][ai][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[0][ai][1]]
                    
                    # aspect[0][0]="(ASPECT "+aspect[0][0]
                    # aspect[0][-1]=aspect[0][-1]+")"
                    
                    aspect[0][0]="("+category+" "+aspect[0][0]
                    
                    aspect[0][-1]=aspect[0][-1]+")"
                    
                    
                    # if "hybrid" in  q[0][ai][0]:
                    #     print(aspect)
                    
                    
                    
                    aspects.append(aspect)
                    
                    if len(sentiment)!=1:
                        opinion_tree_quad_list.append([category,q[0][ai][0],sentiment[ai],q[1][0][0],[q[0][ai][1][0],q[0][ai][1][1]],[q[1][0][1][0],q[1][0][1][1]]])
                    else:
                        opinion_tree_quad_list.append([category,q[0][ai][0],sentiment[0],q[1][0][0],[q[0][ai][1][0],q[0][ai][1][1]],[q[1][0][1][0],q[1][0][1][1]]])
                    
                    
                
                if opinion[1][0]==q[2][0]:
                    opinion[0][0] = "(Q "+opinion[0][0]
                if opinion[1][1]==q[2][1]:
                    opinion[0][-1] = opinion[0][-1]+")"
                for ai in range(len(aspects)):
                    if aspects[ai][1][0]==q[2][0]:
                        aspects[ai][0][0]  = "(Q "+aspects[ai][0][0] 

                    if aspects[ai][1][1]==q[2][1]:
                        aspects[ai][0][-1] = aspects[ai][0][-1]+")"
                    
                tree_q.append(aspects)
                tree_q.append(opinion) 
                tree_quad_list_mA2sO.append(tree_q)
                
            # single aspect to single opinion    
            else:               
                
                if q[1][0][0]=="NULL" and  q[0][0][0]=="NULL":
                    category="_".join([ c.upper() for c in q[0][0][2].split() ])
                    aspect= [["(A "+w+")" for w in q[0][0][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[0][0][1]]
                    # aspect[0][0]="(ASPECT "+aspect[0][0]
                    # aspect[0][-1]=aspect[0][-1]+")"
                    
                    aspect[0][0]="(Q "+"("+category+" "+aspect[0][0]
                    aspect[0][-1]=aspect[0][-1]+")"

                    sentiment=q[1][0][2].upper()
                    opinion= [["(O "+w+")" for w in q[1][0][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[1][0][1]]
                    # opinion[0][0]="(OPINION "+opinion[0][0]
                    # opinion[0][-1]=opinion[0][-1]+")"
                    
                    opinion[0][0]="("+sentiment+" "+opinion[0][0]
                    opinion[0][-1]=opinion[0][-1]+")"+")"
                    
                    tree_q.append(aspect)
                    tree_q.append(opinion) 
                    tree_quad_list_dual_missing.append(tree_q)
                    
                    opinion_tree_quad_list.append([category,q[0][0][0],sentiment,q[1][0][0],[q[0][0][1][0],q[0][0][1][1]],[q[1][0][1][0],q[1][0][1][1]]])
                    
                    
                    
                elif q[1][0][0]=="NULL":
                    
                    category="_".join([ c.upper() for c in q[0][0][2].split() ])
                    sentiment=q[1][0][2].upper()
                    
                    aspect= [["(A "+w+")" for w in q[0][0][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[0][0][1]]
                    # aspect[0][0]="(ASPECT "+aspect[0][0]
                    # aspect[0][-1]=aspect[0][-1]+")"
                    
                    aspect[0][0]="(Q "+"("+sentiment+" "+"("+category+" "+aspect[0][0]
                    aspect[0][-1]=aspect[0][-1]+")"+")"+")"

                    tree_q.append(aspect)
                    tree_quad_list_sA20O.append(tree_q)
                    
                    opinion_tree_quad_list.append([category,q[0][0][0],sentiment,q[1][0][0],[q[0][0][1][0],q[0][0][1][1]],[q[1][0][1][0],q[1][0][1][1]]])
                    
                elif q[0][0][0]=="NULL":
                    category="_".join([ c.upper() for c in q[0][0][2].split() ])
                    sentiment=q[1][0][2].upper()
                    
                    opinion= [["(O "+w+")" for w in q[1][0][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[1][0][1]]
                    # opinion[0][0]="(OPINION "+opinion[0][0]
                    # opinion[0][-1]=opinion[0][-1]+")"
                    
                    opinion[0][0]="(Q "+"("+category+" "+"("+sentiment+" "+opinion[0][0]
                    opinion[0][-1]=opinion[0][-1]+")"+")"+")"

                    tree_q.append(opinion) 
                    tree_quad_list_0A2sO.append(tree_q)
                    
                    opinion_tree_quad_list.append([category,q[0][0][0],sentiment,q[1][0][0],[q[0][0][1][0],q[0][0][1][1]],[q[1][0][1][0],q[1][0][1][1]]])
                 
                else:
                    category="_".join([ c.upper() for c in q[0][0][2].split() ])
                    aspect= [["(A "+w+")" for w in q[0][0][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[0][0][1]]
                    # aspect[0][0]="(ASPECT "+aspect[0][0]
                    # aspect[0][-1]=aspect[0][-1]+")"
                    
                    aspect[0][0]="("+category+" "+aspect[0][0]
                    aspect[0][-1]=aspect[0][-1]+")"

                    sentiment=q[1][0][2].upper()
                    opinion= [["(O "+w+")" for w in q[1][0][0].replace("(","-LRB-").replace(")","-RRB-").split()] , q[1][0][1]]
                    # opinion[0][0]="(OPINION "+opinion[0][0]
                    # opinion[0][-1]=opinion[0][-1]+")"
                    
                    opinion[0][0]="("+sentiment+" "+opinion[0][0]
                    opinion[0][-1]=opinion[0][-1]+")"



                    if opinion[1][0]==q[2][0]:
                        opinion[0][0] = "(Q "+opinion[0][0]
                        aspect[0][-1] = aspect[0][-1]+")"
                    else:
                        aspect[0][0]  = "(Q "+aspect[0][0] 
                        opinion[0][-1] = opinion[0][-1]+")"

                    tree_q.append(aspect)
                    tree_q.append(opinion) 
                    tree_quad_list_sA2sO.append(tree_q)
                    #print(q[0])
                    
                    opinion_tree_quad_list.append([category,q[0][0][0],sentiment,q[1][0][0],[q[0][0][1][0],q[0][0][1][1]],[q[1][0][1][0],q[1][0][1][1]]])
                
                
        #build opinion tree and span tree
        
        offset=2
        for q in tree_quad_list_sA2mO:
            tree[offset+q[0][1][0]:offset+q[0][1][1]]=q[0][0]
            for oi in range(len(q[1])):
                tree[offset+q[1][oi][1][0]:offset+q[1][oi][1][1]]=q[1][oi][0]
        for q in tree_quad_list_sA2sO:
            tree[offset+q[0][1][0]:offset+q[0][1][1]]=q[0][0]
            tree[offset+q[1][1][0]:offset+q[1][1][1]]=q[1][0]
     
        for q in tree_quad_list_mA2sO:
            tree[offset+q[1][1][0]:offset+q[1][1][1]]=q[1][0]
            for ai in range(len(q[0])):
                tree[offset+q[0][ai][1][0]:offset+q[0][ai][1][1]]=q[0][ai][0]
                
        for q in tree_quad_list_sA20O+tree_quad_list_0A2sO:
            tree[offset+q[0][1][0]:offset+q[0][1][1]]=q[0][0]
            
        for q in tree_quad_list_dual_missing:
            
            #if q[0][1]==[-2,-1]:
                
            tree[0:1]=q[0][0]
            #if q[0][1]==[-1,-1]:
            tree[1:2]=q[1][0]
                
        tree="(TOP "+" ".join(tree)+")"
                
        
      
        sentienceDict=dict()
        sentienceDict["sentence"]=i[0]
        sentienceDict["aspect"]=i[1]
        sentienceDict["opinion"]=i[3]
        sentienceDict["aspectSpan"]=i[2]
        sentienceDict["opinionSpan"]=i[4]
        sentienceDict["sentiment"]=i[5]
        sentienceDict["category"]=i[6]
        
        # print(quad_list[0])
        # print(len(quad_list[0]))
        # #print(quad_list[1])
        
        new_aspect=[]
        new_opinion=[]
        new_aspect_span=[]
        new_opinion_span=[]
        new_sentiment=[]
        new_category=[]
        
        
        quad_list_for_structure=[]
        triple_list_for_structure=[]
        pair_list_for_structure=[]
        for c,a,s,o,asp,osp in opinion_tree_quad_list:
            
            new_aspect.append(a)
            new_opinion.append(o)
            new_aspect_span.append(asp)
            new_opinion_span.append(osp)
            new_sentiment.append(s)
            new_category.append(c)
            

            ca="( aspect ( "+" , ".join([c.lower().replace("_"," "),a.lower()])+" ) )"

            so="( opinion ( "+" , ".join([s.lower(),o.lower()])+" ) )"

            quad="( quad "+" , ".join([ca,so])+" )"
            quad_list_for_structure.append(quad)
            
            
            ca_tri="( aspect ( "+a.lower()+" ) )"

            so_tri="( opinion ( "+" , ".join([s.lower(),o.lower()])+" ) )"

            tri="( quad "+" , ".join([ca_tri,so_tri])+" )"
            triple_list_for_structure.append(tri)
            
            
            ca_pair="( aspect ( "+a.lower()+" ) )"

            so_pair="( opinion ( "+o.lower()+" ) )"

            pair="( quad "+" , ".join([ca_pair,so_pair])+" )"
            pair_list_for_structure.append(pair)


                
        final_structures="( root "+" , ".join(quad_list_for_structure)+" )"
        
        final_structures_triple="( root "+" , ".join(triple_list_for_structure)+" )"
        final_structures_pair="( root "+" , ".join(pair_list_for_structure)+" )"
        
        
        
        sentienceDict=dict()
        sentienceDict["sentence"]=i[0]
        sentienceDict["aspect"]=new_aspect
        sentienceDict["opinion"]=new_opinion
        sentienceDict["aspectSpan"]=new_aspect_span
        sentienceDict["opinionSpan"]=new_opinion_span
        sentienceDict["sentiment"]=new_sentiment
        sentienceDict["category"]=new_category
        
        
        sentienceDict["structure"]=final_structures
        sentienceDict["structure_tri"]=final_structures_triple
        sentienceDict["structure_pair"]=final_structures_pair
        sentienceDict["quad_list"]=quad_list
        
        sentienceDict["tree"]=tree
        
        
        
        
        
        
        json_str = json.dumps(sentienceDict,ensure_ascii=False)
        outputFile.write(json_str+"\n")  
        outputFile_only_tree.write(tree+"\n")
        
    print(file)
    
    print("diff:"+str(diff)+" "+str(diff/(blanked+nested+cross+diff+left)))
    print("nested:"+str(nested)+" "+str(nested/(blanked+nested+cross+diff+left)))
    print("cross:"+str(cross)+" "+str(cross/(blanked+nested+cross+diff+left)))
    print("left:"+str(left)+" "+str(left/(blanked+nested+cross+diff+left)))
    print()
    

    
    
    
    outputFile.close()
    outputFile_only_tree.close()
            


# In[5]:


readfile("lap_dev_syn_pos_syn_pos_amr_double_triple_short_deep_2pair_v2.json")
readfile("lap_train_syn_pos_syn_pos_amr_double_triple_short_deep_2pair_v2.json")
readfile("lap_test_syn_pos_syn_pos_amr_double_triple_short_deep_2pair_v2.json")

readfile("res_dev_syn_pos_syn_pos_amr_double_triple_short_deep_2pair_v2.json")
readfile("res_train_syn_pos_syn_pos_amr_double_triple_short_deep_2pair_v2.json")
readfile("res_test_syn_pos_syn_pos_amr_double_triple_short_deep_2pair_v2.json")


