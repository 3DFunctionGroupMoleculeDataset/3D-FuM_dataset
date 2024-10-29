import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import collections
# import SMILESProcessing
import os
import logging
import tqdm
from torch_geometric.datasets import QM9
import rdkit.Chem
from rdkit.Chem import BRICS
from rdkit.Chem import QED
from rdkit.Chem import Descriptors


#这里给出大家注释方便理解
class newQM9Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.aromaticDict={}
        self.pos=[]
        self.rdkitSerial={}
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    # @property
    # def raw_file_names(self):
    #     return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['proccessedData.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        
        # path="./200_300/"
        path="./fullQM9/"
        files=os.listdir(path)
        self.aromaticDict={}
        allFileList=[]
        for file in files:
            allFileList.append((file,file.split("_")[1].split(".")[0]))
        allFileList=sorted(allFileList,key=lambda t:t[1])

        # logging.basicConfig(filename='./datasetRebuildLog/atomNumberError.log', level=logging.INFO) 
        dataset=QM9(root='./dataset/originQM9')
        filePointer=0

        data_list=[]
        atomicNumberDict={'H':1,'C':6,'N':7,'O':8,'F':9}
        for file in tqdm.tqdm(allFileList,desc='Processing'):
            if not os.path.isdir(file[0]):


                self.pos=[]
                self.rdkitSerial={}

                f=open(path+"/"+file[0])

                lines=f.readlines()

                # position 
                self.atomInfoProcess(lines)
                position=torch.tensor(self.pos)



                fileIdx=lines[1].split("\t")[0][4:]
                fileIdxCompare=dataset[filePointer].name[4:]

                if fileIdx==fileIdxCompare:
                    propertyTensor=dataset[filePointer].y
                    x=dataset[filePointer].x
                    self.pos=dataset[filePointer].pos
                    filePointer=filePointer+1
                else:
                    continue



                # properties
                # secondLine=lines[1].split("\t")
                # propertyList=[]
                # propertyList.append(float(secondLine[4]))
                # propertyList.append(float(secondLine[5]))
                # propertyList.append(float(secondLine[6])*27.211396)
                # propertyList.append(float(secondLine[7])*27.211396)
                # propertyList.append(float(secondLine[8])*27.211396)
                # propertyList.append(float(secondLine[9]))
                # propertyList.append(float(secondLine[10])*27.211396)
                # propertyList.append(float(secondLine[11])*27.211396)
                # propertyList.append(float(secondLine[12])*27.211396)
                # propertyList.append(float(secondLine[13])*27.211396)
                # propertyList.append(float(secondLine[14])*27.211396)
                # propertyList.append(float(secondLine[15]))
                # propertyList.append(float(secondLine[1]))
                # propertyList.append(float(secondLine[2]))
                # propertyList.append(float(secondLine[3]))
                # propertyTensor=torch.tensor([propertyList])

                # atomNumber
                atomNumberList=[]
                for line in lines[2:-3]:
                    tempSplit=line.split("\t")[0]
                    atomNumberList.append(atomicNumberDict[tempSplit[0]])
                z=torch.tensor(atomNumberList,dtype=torch.long)
                # SMILES processing

                target=lines[-2].split("\t")[0]

                # returnValue=SMILESProcessing.process(target)
                returnValue=self.SMILESProcessing(target)

                # calculate QED and -logP
                QED_V,LogP_V=self.QED_LogP_calculation(target=target)
                QEDTensor=torch.tensor(QED_V)
                LogPTensor=torch.tensor(LogP_V)

                combination=self.getCombination(target,self.rdkitSerial)
                
                com=torch.tensor(combination)
                # if atom number not match , output to log
                # if len(atomNumberList)!= len(returnValue):
                #     logging.info("atom number not match:"+file+" should get "+str(len(atomNumberList))+" actual get "+str(len(returnValue)))

                edgeIndexList=[[],[]]
                edgeAttrList=[]
                for i in range(0,len(returnValue)):
                    for j in range(0,len(returnValue[i])):
                        edgeIndexList[0].append(i)
                        edgeIndexList[1].append(returnValue[i][j][0])
                        # edgeAttrList.append([returnValue[i][j][1]])
                        edgeAttrValue=[1,0,0,0]
                        if returnValue[i][j][1]==1:
                            edgeAttrValue=[1,0,0,0]
                        elif returnValue[i][j][1]==2:
                            edgeAttrValue=[0,1,0,0]
                        elif returnValue[i][j][1]==3:
                            edgeAttrValue=[0,0,1,0]
                        elif returnValue[i][j][1]==4:
                            edgeAttrValue=[0,0,0,1]
                        # edgeAttrValueTensor=torch.tensor(edgeAttrValue)
                        # edgeAttrList.append(edgeAttrValueTensor)
                        edgeAttrList.append(edgeAttrValue)
                # print(edgeIndexList)
                # print(edgeAttrList)
                edge_index=torch.tensor(edgeIndexList,dtype=torch.long)
                edge_attr=torch.tensor(edgeAttrList,dtype=torch.long)
                # print(edge_index)
                # print(edge_attr)


                # print("target:",target)

                f.close()
                data = Data(x=x,y=propertyTensor,z=z, edge_index=edge_index, edge_attr=edge_attr,pos=self.pos,com=com,QED=QEDTensor,LogP=LogPTensor)
                # put into datalist
                data_list.append(data)
        print(len(self.aromaticDict))


 

        
        

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print(len(data_list))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def SMILESProcessing(self,text):
        tempList=[]
        bracketMatchDict={}
        atomDict={}
        ionicDict={'[NH3':([1,2,3,4,5],4),'[NH2':([1,2,3,4,5],3),'[NH+':([1,2,3,4],2),'[O-]':([1,2,3],1),'[N-]':([1,2,3],1),'[CH-':([1,2,3,4],2),'[C-]':([1,2,3],1)}
        jumpList=[]
        middleBracketMatchDict={}
        tempRingDict=collections.defaultdict(list)
        ringMatchDict={}
        atomCount=0
        p=-1
        for (index,i) in  enumerate(text)   :
            # if index in jump list
            if index in jumpList:
                continue

            # if it is ionic
            if i=='[':
                ionicValue=ionicDict[text[index:index+4]]
                for j in ionicValue[0]:
                    jumpList.append(index+j)
                # print(ionicValue[0][-1])
                middleBracketMatchDict[index]=index+ionicValue[0][-1]
                middleBracketMatchDict[index+ionicValue[0][-1]]=index
                atomDict[index+1]=atomCount
                atomCount=atomCount+1
                ionicPose=[]
                # print(ionicValue[1]-1)
                if ionicValue[1]-1 >0:
                    for j in range(ionicValue[1]-1):
                        ionicPose.append(atomCount)
                        atomCount=atomCount+1
                    atomDict[index+2]=ionicPose


            # QM9 dataset only contain C,N,O,F,H, H will not show in SMILES
            if i=='C' or i=='N' or i=='O' or i=='F' or i=='c' or i=='n' or i=='o' or i=='f':
                atomDict[index]=atomCount
                atomCount=atomCount+1
                continue
            # Restricted to molecular weight, no individual data in the QM9 dataset will have more than 9 rings
            if i=='1' or i=='2' or i=='3' or i=='4' or i=='5' or i=='6' or i=='7' or i=='8' or  i=='9':
                if len(tempRingDict[i])==0:
                    tempRingDict[i].append(index)
                else:
                    ringMatchDict[tempRingDict[i][0]]=index
                    ringMatchDict[index]=tempRingDict[i][0]
                    tempRingDict[i].pop()
                continue
            # bracket match
            if i=='(':
                tempList.append(index)
                # p=len(tempList)-1
                continue
            if i==')':
                # while(tempList[p][1]!=-1):
                #     if tempList[p][1]==-1:
                #         break
                #     p=p-1
                # tempList[p][1]=index

                bracketMatchDict[tempList[-1]]=index
                tempList.pop()
                continue
        # Hpointer=len(atomDict)
        Hpointer=atomCount
        # use tuple to store edge info, (connectionNode,edgeAttr)
        edgeInfo=collections.defaultdict(list)
        # C with 4 bonds, N with 3, O with 2, H/F with 1
        bondTotal={'C':4,'c':4,'N':3,'n':3,'O':2,'o':2,'F':1,'f':1}


        # will exist atom after ')', may not exist atom after number

        # The order of the SMILES representation starts with rings, then branches, and finally directly connected atoms. The point where the ring cuts through will only be a single bond
        for (index,i) in  enumerate(text)   :
            # print(index)
            if index in jumpList:
                continue
            # ionic process
            if i=='[':
                next=middleBracketMatchDict[index]+1
                # H connection
                if (index+2) in atomDict:
                    for j in atomDict[index+2]:
                        edgeInfo[atomDict[index+1]].append((j,1))
                        edgeInfo[j].append((atomDict[index+1],1))

                while next<len(text):

                    if text[next]=='[':
                        edgeInfo[atomDict[index+1]].append((atomDict[next+1],1))
                        edgeInfo[atomDict[next+1]].append((atomDict[index+1],1))
                        break

                    if text[next]=='C' or text[next]=='N' or text[next]=='O' or text[next]=='F' or text[next]=='c' or text[next]=='n' or text[next]=='o' or text[next]=='f':
                        # print(index,atomDict[index],next,atomDict[next],edgeInfo[atomDict[index]])
                        edgeInfo[atomDict[index+1]].append((atomDict[next],1))
                        edgeInfo[atomDict[next]].append((atomDict[index+1],1))
                        # print(edgeInfo[atomDict[index]])
                        break
                    if text[next]=='=':
                        # if next is ionic
                        if text[next+1]=='[':
                            edgeInfo[atomDict[index+1]].append((atomDict[next+2],2))
                            edgeInfo[atomDict[next+2]].append((atomDict[index+1],2))
                        else:
                            edgeInfo[atomDict[index+1]].append((atomDict[next+1],2))
                            edgeInfo[atomDict[next+1]].append((atomDict[index+1],2))
                        break
                    if text[next]=='#':
                        if text[next+1]=='[':
                            edgeInfo[atomDict[index+1]].append((atomDict[next+2],3))
                            edgeInfo[atomDict[next+2]].append((atomDict[index+1],3))
                        else:
                            edgeInfo[atomDict[index+1]].append((atomDict[next+1],3))
                            edgeInfo[atomDict[next+1]].append((atomDict[index+1],3))
                        break
                    if text[next].isdigit()==True:
                        if ringMatchDict[next]<next:
                            next=next+1
                            continue
                        elif ringMatchDict[next]>next:
                            ringAtomIndex=ringMatchDict[next]-1
                            while text[ringAtomIndex].isdigit()==True:
                                if text[ringAtomIndex].isdigit()==False:
                                    break
                                ringAtomIndex=ringAtomIndex-1
                            if text[ringAtomIndex]==']':
                                leftMiddBracket=middleBracketMatchDict[ringAtomIndex]
                                edgeInfo[atomDict[index+1]].append((atomDict[leftMiddBracket+1],1))
                                edgeInfo[atomDict[leftMiddBracket+1]].append((atomDict[index+1],1))
                            else:
                                while ringAtomIndex not in atomDict:
                                    if ringAtomIndex in atomDict:
                                        break
                                    ringAtomIndex=ringAtomIndex-1
                                edgeInfo[atomDict[index+1]].append((atomDict[ringAtomIndex],1))
                                edgeInfo[atomDict[ringAtomIndex]].append((atomDict[index+1],1))
                            next=next+1
                            continue
                    if text[next]=='(':
                        if text[next+1]=='=':
                            if text[next+2]=='[':
                                edgeInfo[atomDict[index+1]].append((atomDict[next+3],2))
                                edgeInfo[atomDict[next+3]].append((atomDict[index+1],2))
                            else:
                                edgeInfo[atomDict[index+1]].append((atomDict[next+2],2))
                                edgeInfo[atomDict[next+2]].append((atomDict[index+1],2))

                            next=bracketMatchDict[next]
                            next=next+1
                            continue
                        if text[next+1]=='#':
                            if text[next+2]=='[':
                                edgeInfo[atomDict[index+1]].append((atomDict[next+3],3))
                                edgeInfo[atomDict[next+3]].append((atomDict[index+1],3))
                            else:
                                edgeInfo[atomDict[index+1]].append((atomDict[next+2],3))
                                edgeInfo[atomDict[next+2]].append((atomDict[index+1],3))

                            next=bracketMatchDict[next]
                            next=next+1
                            continue
                        if text[next+1]=='[':
                            edgeInfo[atomDict[index+1]].append((atomDict[next+2],1))
                            edgeInfo[atomDict[next+2]].append((atomDict[index+1],1))
                            next=bracketMatchDict[next]
                            next=next+1
                            continue
                        if (next+1) in atomDict:
                            edgeInfo[atomDict[index+1]].append((atomDict[next+1],1))
                            edgeInfo[atomDict[next+1]].append((atomDict[index+1],1))
                            next=bracketMatchDict[next]
                            next=next+1
                            continue
                    if text[next]==')':
                        break

            # QM9 dataset only contain C,N,O,F,H, H will not show in SMILES
            if i=='C' or i=='N' or i=='O' or i=='F' or i=='c' or i=='n' or i=='o' or i=='f':
                next=index+1
                # print(index)
                while next<len(text):

                    if text[next]=='[':
                        edgeInfo[atomDict[index]].append((atomDict[next+1],1))
                        edgeInfo[atomDict[next+1]].append((atomDict[index],1))
                        break

                    if text[next]=='C' or text[next]=='N' or text[next]=='O' or text[next]=='F' or text[next]=='c' or text[next]=='n' or text[next]=='o' or text[next]=='f':
                        # print(index,atomDict[index],next,atomDict[next],edgeInfo[atomDict[index]])
                        edgeInfo[atomDict[index]].append((atomDict[next],1))
                        edgeInfo[atomDict[next]].append((atomDict[index],1))
                        # print(edgeInfo[atomDict[index]])
                        break
                    if text[next]=='=':
                        # if next is ionic
                        if text[next+1]=='[':
                            edgeInfo[atomDict[index]].append((atomDict[next+2],2))
                            edgeInfo[atomDict[next+2]].append((atomDict[index],2))
                        else:
                            edgeInfo[atomDict[index]].append((atomDict[next+1],2))
                            edgeInfo[atomDict[next+1]].append((atomDict[index],2))
                        break
                    if text[next]=='#':
                        if text[next+1]=='[':
                            edgeInfo[atomDict[index]].append((atomDict[next+2],3))
                            edgeInfo[atomDict[next+2]].append((atomDict[index],3))
                        else:
                            edgeInfo[atomDict[index]].append((atomDict[next+1],3))
                            edgeInfo[atomDict[next+1]].append((atomDict[index],3))
                        break
                    if text[next].isdigit()==True:
                        if ringMatchDict[next]<next:
                            next=next+1
                            continue
                        elif ringMatchDict[next]>next:
                            ringAtomIndex=ringMatchDict[next]-1
                            while text[ringAtomIndex].isdigit()==True:
                                if text[ringAtomIndex].isdigit()==False:
                                    break
                                ringAtomIndex=ringAtomIndex-1



                            if text[ringAtomIndex]==']':
                                leftMiddBracket=middleBracketMatchDict[ringAtomIndex]
                                edgeInfo[atomDict[index]].append((atomDict[leftMiddBracket+1],1))
                                edgeInfo[atomDict[leftMiddBracket+1]].append((atomDict[index],1))
                            else:
                                while ringAtomIndex not in atomDict:
                                    if ringAtomIndex in atomDict:
                                        break
                                    ringAtomIndex=ringAtomIndex-1
                                edgeInfo[atomDict[index]].append((atomDict[ringAtomIndex],1))
                                edgeInfo[atomDict[ringAtomIndex]].append((atomDict[index],1))
                            next=next+1
                            continue
                    if text[next]=='(':
                        if text[next+1]=='=':
                            if text[next+2]=='[':
                                edgeInfo[atomDict[index]].append((atomDict[next+3],2))
                                edgeInfo[atomDict[next+3]].append((atomDict[index],2))
                            else:
                                edgeInfo[atomDict[index]].append((atomDict[next+2],2))
                                edgeInfo[atomDict[next+2]].append((atomDict[index],2))

                            next=bracketMatchDict[next]
                            next=next+1
                            continue
                        if text[next+1]=='#':
                            if text[next+2]=='[':
                                edgeInfo[atomDict[index]].append((atomDict[next+3],3))
                                edgeInfo[atomDict[next+3]].append((atomDict[index],3))
                            else:
                                edgeInfo[atomDict[index]].append((atomDict[next+2],3))
                                edgeInfo[atomDict[next+2]].append((atomDict[index],3))

                            next=bracketMatchDict[next]
                            next=next+1
                            continue
                        if text[next+1]=='[':
                            edgeInfo[atomDict[index]].append((atomDict[next+2],1))
                            edgeInfo[atomDict[next+2]].append((atomDict[index],1))
                            next=bracketMatchDict[next]
                            next=next+1
                            continue
                        if (next+1) in atomDict:
                            edgeInfo[atomDict[index]].append((atomDict[next+1],1))
                            edgeInfo[atomDict[next+1]].append((atomDict[index],1))
                            next=bracketMatchDict[next]
                            next=next+1
                            continue
                    if text[next]==')':
                        break

                bondCount=0
                HConnection=0
                for j in edgeInfo[atomDict[index]]:
                    bondCount=bondCount+j[1]
                HConnection=bondTotal[i]-bondCount
                for k in range(0,HConnection):
                    edgeInfo[atomDict[index]].append((Hpointer,1))
                    edgeInfo[Hpointer].append((atomDict[index],1))
                    Hpointer=Hpointer+1
        return edgeInfo
    def atomInfoProcess(self,lines):
        counter=0
        for  idx,i in  enumerate(lines[2:-3]) :
            tempPos=[]
            iSplit=i.split("\t")
            atom=iSplit[0]
            # tempPos.append(float(iSplit[1]))
            # tempPos.append(float(iSplit[2]))
            # tempPos.append(float(iSplit[3]))
            self.pos.append(tempPos)
            if atom!='H':
                self.rdkitSerial[counter]=idx
                counter=counter+1
    def checkaromaticDict(self,aList):
        tempList=collections.deque(aList)
        flag=False
        for i in range(len(aList)):
            if tuple(tempList) in self.aromaticDict:
                return self.aromaticDict[tuple(tempList)]
            temp=tempList.popleft()
            tempList.append(temp)
        if flag==False:
            self.aromaticDict[tuple(tempList)]=len(self.aromaticDict)+1000
            return self.aromaticDict[tuple(tempList)]
    def checkDTBond(self,beginAtom,endAtom,bond):
        if beginAtom>=endAtom:
            return beginAtom*100+bond*10+endAtom
        else:
            return beginAtom+bond*10+endAtom*100
    def getCombination(self,text,rdkitSerial):
        com=[]
        tempList=[]
        mol=rdkit.Chem.MolFromSmiles(text)
        aromaticList=mol.GetAromaticAtoms()

        aromaticPosList=[]
        for aromaticAtom in aromaticList:
            aromaticPosList.append(aromaticAtom.GetIdx())

        aromaticLen=len(aromaticList)

        # if aromaticLen==0:
        #     # print("no aromatic")
        # else:
        if aromaticLen!=0:
            for atomidx,atom in  enumerate(aromaticList)   :
                tempList.append(atom)
                # print(tempList)
                if atomidx+1 != aromaticLen:
                    if mol.GetBondBetweenAtoms(aromaticList[atomidx].GetIdx(),aromaticList[atomidx+1].GetIdx())!=None and mol.GetBondBetweenAtoms(aromaticList[atomidx].GetIdx(),aromaticList[atomidx+1].GetIdx()).GetIsAromatic()==False:
                        posList=[]
                        aList=[]
                        for i in tempList:
                            posList.append( rdkitSerial[i.GetIdx()]  )
                            aList.append(i.GetAtomicNum())
                        aromaticResult=self.checkaromaticDict(aList)



# TODO: Unify the length of the combination. Makes it possible to make them into tensors.

                        # posList.append(aromaticResult)

                        # posStr=','.join(posList)
                        for posI in range(0,9-len(posList)):
                            posList.append(-1)
                        posList.append(aromaticResult)
                        com.append(posList)
                        # com.append((posList,aromaticResult))
                        tempList=[]
                else:
                    posList=[]
                    aList=[]
                    for i in tempList:
                        posList.append( rdkitSerial[i.GetIdx()]  )
                        aList.append(i.GetAtomicNum())
                    aromaticResult=self.checkaromaticDict(aList)
                    for posI in range(0,9-len(posList)):
                            posList.append(-1)
                    posList.append(aromaticResult)
                    com.append(posList)
                    # com.append((posList,aromaticResult))
                    # tempList=[]
        for bond in mol.GetBonds():

            if bond.GetBeginAtom().GetIdx() in aromaticPosList or bond.GetEndAtom().GetIdx() in aromaticPosList:
                continue

            if bond.GetBondType()==rdkit.Chem.rdchem.BondType.DOUBLE  :
                ba=bond.GetBeginAtom()
                ea=bond.GetEndAtom()
                posList=[]
                posList.append(rdkitSerial[ba.GetIdx()] )
                posList.append(rdkitSerial[ea.GetIdx()] )
                dtResult=self.checkDTBond(ba.GetAtomicNum(),ea.GetAtomicNum(),2)
                for posI in range(0,9-len(posList)):
                            posList.append(-1)
                posList.append(dtResult)
                com.append(posList)


                # com.append((posList,dtResult))
            if bond.GetBondType()==rdkit.Chem.rdchem.BondType.TRIPLE:
                ba=bond.GetBeginAtom()
                ea=bond.GetEndAtom()
                posList=[]
                posList.append(rdkitSerial[ba.GetIdx()] )
                posList.append(rdkitSerial[ea.GetIdx()] )
                dtResult=self.checkDTBond(ba.GetAtomicNum(),ea.GetAtomicNum(),3)
                for posI in range(0,9-len(posList)):
                            posList.append(-1)
                posList.append(dtResult)
                com.append(posList)

                # com.append((posList,dtResult))
        return com
    
    # QED calculation
    def QED_LogP_calculation(self,target):
        mol=rdkit.Chem.MolFromSmiles(target)
        QED_V=QED.qed(mol)
        LogP_V=Descriptors.MolLogP(mol)
        return (QED_V,LogP_V)

# b=newQM9Dataset("QM9_200_300")
b=newQM9Dataset("newQM9_20231228")
print(len(b))
print(b[0])
print(b[1])
print(len(b.aromaticDict))

# """测试"""
# b = MyOwnDataset("MYdata")

# >>>Process
# b.data.num_features
# >>>1
# b.data.num_nodes
# >>>3
# b.data.num_edges
# >>>4