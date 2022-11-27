# CCFs_ LWP_ SWfb-linear-regression
Build a rergression model for some Cloud metrics(like LWP or sth ..) to picked CCF(Cloud Controlling Factors) in GCMs .


# Nov 27 updated
Cloud metric choosed: Liquid Water Path ('clwvi' - 'clivi');  CCF: Srface Temperature ('ts'), Precipitation - Evaporation ('pr', 'hfls') (Need to to prove they are close to Moisture Convergence),
Subsidence at 500 mb ('wap' at 500 hPa) , and LTS (Lower Tropospheric Stability = (the Potential Temperature at 700 mb - Surface), 'ps', 'ts', 'ta' at 700 hPa). 
Using some modules to help analyze the GCM and Observational data.
