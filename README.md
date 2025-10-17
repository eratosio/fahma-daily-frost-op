# fahma-daily-frost-op
FAHMA daily frost metrics on demand operator


# Deployment

build/deploy script. The steps are roughly:

### Create model arhcive

Create zip file with everything in `src`

### Upload zip to senaps

Upload via https://senaps.eratos.com/#/app/model/upload

### Create Operator and Block resources

User `pushprivateblocks.py` (NOTE this hasn't been reviewed for a while so unclear on state)


### Testing

There is a google colab notebook which can be used for testing end-to-end:

https://colab.research.google.com/drive/1Ig8-4CMLIH-jZKufNsMA89crjWTh7-A9?usp=sharing#scrollTo=G0KAQJoWk3Uu