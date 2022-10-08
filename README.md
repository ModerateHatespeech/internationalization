# Internationalization API

Support for multiple languages is provided by a many-to-one Translator model responsible for converting detected languages to English for detection.

[OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en) is used for language translation, implemented via HuggingFace providing support for the following language codes:

>    abk acm ady afb afh_Latn afr akl_Latn aln amh ang_Latn apc ara arg arq ary arz asm ast avk_Latn awa aze_Latn bak bam_Latn bel bel_Latn ben bho bod bos_Latn bre brx brx_Latn bul bul_Latn cat ceb ces cha che chr chv cjy_Hans cjy_Hant cmn cmn_Hans cmn_Hant cor cos crh crh_Latn csb_Latn cym dan deu dsb dtp dws_Latn egl ell enm_Latn epo est eus ewe ext fao fij fin fkv_Latn fra frm_Latn frr fry fuc fuv gan gcf_Latn gil gla gle glg glv gom gos got_Goth grc_Grek grn gsw guj hat hau_Latn haw heb hif_Latn hil hin hnj_Latn hoc hoc_Latn hrv hsb hun hye iba ibo ido ido_Latn ike_Latn ile_Latn ilo ina_Latn ind isl ita izh jav jav_Java jbo jbo_Cyrl jbo_Latn jdt_Cyrl jpn kab kal kan kat kaz_Cyrl kaz_Latn kek_Latn kha khm khm_Latn kin kir_Cyrl kjh kpv krl ksh kum kur_Arab kur_Latn lad lad_Latn lao lat_Latn lav ldn_Latn lfn_Cyrl lfn_Latn lij lin lit liv_Latn lkt lld_Latn lmo ltg ltz lug lzh lzh_Hans mad mah mai mal mar max_Latn mdf mfe mhr mic min mkd mlg mlt mnw moh mon mri mwl mww mya myv nan nau nav nds niu nld nno nob nob_Hebr nog non_Latn nov_Latn npi nya oci ori orv_Cyrl oss ota_Arab ota_Latn pag pan_Guru pap pau pdc pes pes_Latn pes_Thaa pms pnb pol por ppl_Latn prg_Latn pus quc qya qya_Latn rap rif_Latn roh rom ron rue run rus sag sah san_Deva scn sco sgs shs_Latn shy_Latn sin sjn_Latn slv sma sme smo sna snd_Arab som spa sqi srp_Cyrl srp_Latn stq sun swe swg swh tah tam tat tat_Arab tat_Latn tel tet tgk_Cyrl tha tir tlh_Latn tly_Latn tmw_Latn toi_Latn ton tpw_Latn tso tuk tuk_Latn tur tvl tyv tzl tzl_Latn udm uig_Arab uig_Cyrl ukr umb urd uzb_Cyrl uzb_Latn vec vie vie_Hani vol_Latn vro war wln wol wuu xal xho yid yor yue yue_Hans yue_Hant zho zho_Hans zho_Hant zlm_Latn zsm_Latn zul zza

## Implementation

Internationalization may be implemented by calling `/api/v1/intl/<endpoint>`, where <endpoint> is the desired model endpoint. 
  
For example, to call `https://api.moderatehatespeech.com/api/v1/moderate/` with support for multiple languages, use the `https://api.moderatehatespeech.com/api/v1/intl/moderate/` endpoint.
  
Requests are first passed to the hosted translation model (served via Flask/Gunicorn) before being passed to the original endpoint. 
  
## Usage
  
To host your own translation endpoint for language preprocessing, clone the repository and build the docker image using:
  
`docker build . -t mhs_translate`
  
Then, run the image (pass `--gpus all` for CUDA support) which will be available on port 8081
  
## Accuracy

OPUS-MT, while generally robust, comes with some accuracy drawbacks. For more powerful CUDA devices, upping the num_beams parameter (to ~30-40) can slightly increas accuracy. However, future deployments could investigate M2M, mBart50, and ISI's RTG v2.1 model.

While inaccuracies in translation will result in a perfomance loss on prediction, generalize capture of a message's meaning is usually suitable for predictions. 
