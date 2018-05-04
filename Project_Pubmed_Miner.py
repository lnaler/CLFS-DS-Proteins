# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:33:20 2018

@author: lbnal
"""
from Bio import Entrez
from matplotlib import pyplot as plt

def search(terms):
    Entrez.email = 'lnaler@vt.edu'
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance', 
                            retmax='20',
                            retmode='xml',
                            idtype="acc",
                            term=terms)
    results = Entrez.read(handle)
    handle.close()
    return results

#ideally, this would be combined with search using history
def get_papers(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'lnaler@vt.edu'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=id_list)
    results = Entrez.read(handle)
    handle.close()
    return results

#We are going to pull the data from NCBI PubMed and dump it in a text file and pie chart
def mine_PubMed(save_file, tree_search, pie_file, title):
    f = open(save_file,'w', encoding="utf-8")
    f.write("ANALYSIS OF FEATURES\n")
    paper_count = {}
    for term in tree_search:
        print("Currently searching for: ", term)
        search_term = "("+term+") AND Down Syndrome"
        f.write("===============================================================\n")
        f.write("\nSearch term: %s \n" % term)
        results = search(search_term)
        id_list = results['IdList']
        if(len(id_list) > 0):
            papers = get_papers(id_list)
            for i, paper in enumerate(papers):
                temp_list = papers[paper]
                size = len(temp_list)
                if(i == 0):
                    paper_count[term] = size #care only about articles
                #paper_count[term] = size
                for j in range(size):
                    output = str(j+1) + ": " + temp_list[j]['MedlineCitation']['Article']['ArticleTitle']+"\n"
                    f.write(output)
        else:
            f.write("No records found.\n")
            paper_count[term] = 0
    f.close()
    
    #Let's toss it into a Pie Chart
    num_papers = list(paper_count.values())
    twenty_plus = [i for i in num_papers if i == 20]
    one_ten = [i for i in num_papers if i > 0 and i < 11]
    eleven_twenty = [i for i in num_papers if i > 10 and i < 20]
    zero = [i for i in num_papers if i == 0]
    
    labels = '20+ Records', '11-20 Records', '1-10 Records', '0 Records'
    sizes = [len(twenty_plus), len(eleven_twenty), len(one_ten), len(zero)]
    
    fig1, ax1 = plt.subplots()
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.setp(autotexts, size=12, weight="bold")
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(title)
    plt.show()
    plt.savefig(pie_file)



#Decision Tree
tree_search = ['DYRK1A', 'NR1', 'pAKT', 'pCAMKII', 'pERK', 'pMEK',
               'pPKCAB', 'BRAF', 'APP', 'SOD1', 'MTOR', 'AMPKA',
               'pNUMB', 'TIAM1', 'pP70S6', 'pPKCG', 'S6', 'BAX',
               'ARC', 'ERBB4', 'nNOS', 'Tau', 'IL1B', 'pCASP9',
               'PSD95', 'Ubiquitin', 'pGSK3B_Tyr216']
mine_PubMed('tree_features_pubmed.txt', tree_search, 'tree_features_pubmed.png', 'Decision Tree Features Pubmed Results')

#ANN
ann_search = ['DYRK1A', 'BDNF', 'NR1', 'NR2A', 'pAKT', 'pCAMKII', 'pCREB',
              'pELK', 'pERK', 'pJNK', 'pNR1', 'pNR2A', 'pPKCAB', 'pRSK',
              'AKT', 'BRAF', 'CAMKII', 'CREB', 'ELK', 'ERK', 'JNK', 'MEK',
              'TRKA', 'RSK', 'APP', 'SOD1', 'MTOR', 'P38', 'pMTOR', 'DSCR1',
              'AMPKA', 'NR2B', 'RAPTOR', 'TIAM1', 'P70S6_N', 'CDK5', 'S6',
              'ADARB1', 'BAX', 'nNOS', 'Tau', 'GluR3', 'pCASP9', 'PSD95',
              'SNCA', 'pGSK3B_Tyr216', 'SHH', 'BCL2', 'CaNA']
mine_PubMed('ann_features_pubmed.txt', ann_search, 'ann_features_pubmed.png', 'ANN Features Pubmed Results')

#KMeans
kmeans_search = ['pCAMKII', 'pERK', 'PKCA', 'pMEK', 'pNR2A', 'BRAF',
                 'SOD1', 'pNUMB', 'pPKCG', 'ERBB4', 'Tau', 'GFAP',
                 'Ubiquitin', 'BCL2', 'CaNA']
mine_PubMed('kmeans_features_pubmed.txt', kmeans_search, 'kmeans_features_pubmed.png', 'K-Means Features Pubmed Results')

#Average KMeans
avg_kmeans_search = ['DYRK1A', 'pBRAF', 'pCAMKII', 'PKCA', 'pNR2A',
                     'pPKCAB', 'pRSK', 'BRAF', 'CAMKII', 'ELK', 'MEK',
                     'APP', 'Bcatenin', 'MTOR', 'DSCR1', 'AMPKA', 'TIAM1',
                     'pP70S6', 'P70S6', 'pPKCG', 'CDK5', 'GluR4', 'pS6',
                     'pCFOS', 'H3MeK4', 'CaNA']

mine_PubMed('avg_kmeans_features_pubmed.txt', avg_kmeans_search, 'avg_kmeans_features_pubmed.png', 'Average K-Means Features Pubmed Results')

#Random Forest
forest_search = ['DYRK1A', 'ITSN1', 'BDNF', 'pCAMKII', 'pERK', 'pJNK', 'PKCA',
                 'pMEK', 'pNR2A', 'pPKCAB', 'pRSK', 'AKT', 'BRAF', 'CAMKII', 'APP',
                 'SOD1', 'MTOR', 'P38', 'pMTOR', 'DSCR1', 'AMPKA', 'NR2B', 'pNUMB',
                 'RAPTOR', 'TIAM1', 'pP70S6', 'P70S6', 'pGSK3B', 'pPKCG', 'S6',
                 'ADARB1', 'AcetylH3K9', 'BAX', 'ARC', 'ERBB4', 'nNOS', 'Tau',
                 'GluR3', 'GluR4', 'IL1B', 'P3525', 'pCASP9', 'PSD95', 'SNCA',
                 'Ubiquitin', 'SHH', 'BAD', 'BCL2', 'pS6', 'SYP', 'H3AcK18',
                 'H3MeK4', 'CaNA']

mine_PubMed('forest_features_pubmed.txt', forest_search, 'forest_features_pubmed.png', 'Forest Features Pubmed Results')


#Checking for the intersection between a few feature sets
tree_set = set(tree_search)
ann_set = set(ann_search)
forest_set = set(forest_search)
tree_forest_overlap = tree_set.intersection(forest_set)
tree_forest_ANN_overlap = tree_forest_overlap.intersection(ann_set)


mine_PubMed('tree_forest_overlap.txt', tree_forest_overlap, 'tree_forest_overlap.png', 'Shared Tree and Forest Features')
mine_PubMed('tree_ann_forest_overlap.txt', tree_forest_ANN_overlap, 'tree_forest_ann_overlap.png', 'Shared Tree, Forest, and ANN Features')
