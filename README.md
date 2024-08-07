# nanomamba
The simplest, fastest repository for training/finetuning medium-sized Mambas.

## quick start

```sh
python data/siir/prepare.py
```

this creates the data, in this case its turkish poems.

```sh
python train.py
```

and to inference model after training;

```sh
python sample.py
```

this outputs turkish poems:

```
Bir şiirler gibi ağlar yasılgın değen<br/>
Yakılmış her tarih yüreğim gibiydi<br/>
Kimdi yaralım dudaklarına konuklar, kızılmıştı<br/>
bir külyüllerim ses<br/>
Bir kabusir mardı maksıyordu parzak sana<br/>
yapıldı üstünde köransın<br/>
görmediği gibi seriğinde düşünü böyle<br/>
Baş’i gelir kadınız kadının<br/>
Aydılığımızın dünya yalnızlığın<br/>
Bir tepin ağlımız karı gözlerinde vazit korkuk anlı'<br/>
Kim karşılıkları yokluğum bir dünya<br/>
Belki de uçan ukarları şûhınız<br/>
Şarağız bakı ölmüştür<br/>
Kırılladım kıyımızı sârâyla kırılır<br/>
kocamanlaşikte yapmamış üstünde<br/>
Ölüm bir bir yarması karşırdığınız<br/>
Gözü gibi benin bir kim daha bir gözleri<br/>
O zamanlar kimlik durmasına<br/>
IRüzgarısın kaldı<br/>
Yırtırın düştük kılıtlığını<br/>
```
