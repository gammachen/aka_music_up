from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, BNode
from rdflib.collection import Collection
from owlrl import DeductiveClosure, OWLRL_Extension

ex = Namespace("http://example.org/")
g = Graph()
g.bind("ex", ex)

# 1. 类层次推理 (Subsumption)
g.add((ex.TechCompany, RDFS.subClassOf, ex.Company))
g.add((ex.AICompany, RDFS.subClassOf, ex.TechCompany))

# 2. 属性特征推理 (Property Characteristics)
# 传递性
controls = ex.controls
g.add((controls, RDF.type, OWL.TransitiveProperty))
g.add((ex.CompanyA, controls, ex.CompanyB))
g.add((ex.CompanyB, controls, ex.CompanyC))
g.add((ex.CompanyC, controls, ex.CompanyD))
g.add((ex.CompanyD, controls, ex.CompanyE))
g.add((ex.CompanyE, controls, ex.CompanyF))

# 对称性
partner = ex.partner
g.add((partner, RDF.type, OWL.SymmetricProperty))
g.add((ex.CompanyA, partner, ex.CompanyB))

# 3. 属性链推理 (Property Chains)
g.add((ex.ownsFactory, OWL.propertyChainAxiom, BNode()))
owns_chain = BNode()
Collection(g, owns_chain, [ex.ownsLand, ex.hasFactory])
g.add((ex.ownsFactory, OWL.propertyChainAxiom, owns_chain))
g.add((ex.CompanyA, ex.ownsLand, ex.LandParcel))
g.add((ex.LandParcel, ex.hasFactory, ex.FactoryX))

# 4. 个体分类 (Individual Classification)
g.add((ex.MonopolyCompany, RDF.type, OWL.Class))
# MonopolyCompany ≡ marketShare hasValue 0.5
mono_restriction = BNode()
g.add((mono_restriction, RDF.type, OWL.Restriction))
g.add((mono_restriction, OWL.onProperty, ex.marketShare))
g.add((mono_restriction, OWL.hasValue, Literal(0.5)))
g.add((ex.MonopolyCompany, OWL.equivalentClass, mono_restriction))
g.add((ex.CompanyB, ex.marketShare, Literal(0.5)))

# 5. 一致性检测 (Consistency Checking)
g.add((ex.Company, OWL.disjointWith, ex.Individual))
g.add((ex.CompanyA, RDF.type, ex.Company))
g.add((ex.CompanyA, RDF.type, ex.Individual))

print("=== 类层次推理 ===")
for s, o in g.subject_objects(RDFS.subClassOf):
    print(f"{s.split('/')[-1]} rdfs:subClassOf {o.split('/')[-1]}")

print("执行OWL RL推理...")
DeductiveClosure(OWLRL_Extension).expand(g)
print("推理完成!\n")

# 1. 类层次推理
print("=== 类层次推理 ===")
for s, o in g.subject_objects(RDFS.subClassOf):
    print(f"{s.split('/')[-1]} rdfs:subClassOf {o.split('/')[-1]}")

# 2. 属性特征推理
print("\n=== 传递性推理 ===")
for o in g.objects(ex.CompanyA, controls):
    print(f"CompanyA controls {o.split('/')[-1]}")

print("\n=== 传递性推理-2 ===")
for o in g.objects(ex.CompanyC, controls):
    print(f"CompanyC controls {o.split('/')[-1]}")

print("\n=== 对称性推理 ===")
for o in g.objects(ex.CompanyB, partner):
    print(f"CompanyB partner {o.split('/')[-1]}")

# 3. 属性链推理
print("\n=== 属性链推理 ===")
for o in g.objects(ex.CompanyA, ex.ownsFactory):
    print(f"CompanyA ownsFactory {o.split('/')[-1]}")

# 4. 个体分类
print("\n=== 个体分类推理 ===")
for s in g.subjects(RDF.type, ex.MonopolyCompany):
    print(f"MonopolyCompany: {s.split('/')[-1]}")

# 5. 一致性检测
print("\n=== 一致性检测 ===")
for s, o in g.subject_objects(OWL.disjointWith):
    print(f"Disjoint: {s.split('/')[-1]} 与 {o.split('/')[-1]}")
for s in g.subjects(RDF.type, ex.Company):
    if (s, RDF.type, ex.Individual) in g:
        print(f"{s.split('/')[-1]} 同时属于 Company 和 Individual（应为不一致）")
