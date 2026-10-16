import requests
import sys

url = "https://football.fantasysports.yahoo.com/f1/772025/draft"
cookie_str = """PH=l=en-US; Y=v=1&n=6vev0qcqsvmnp&l=2h0fe20bofje/o&p=m2g2uhf00000000&r=qb&intl=us; cmp=t=1788719627&j=0&u=1YNN; GUC=AQEACAJqlL5qzUIW-QNT&s=AQAAAKAjuBJo&g=apNzzg; ucs=tr=1788830100000; SPTB=d=CM_r_u6wjB2gIcDKnM.jR1kA6jAxJwu37vD2w28Dvi.3oQ8mdThyigQsb7IccwDzGf.SeGUgLmsB0g6aBE9IT.0cEsiR7D8_nmBzIRwv6CU-&v=1; apeaf=userCategoryPrefs%3D%7B%22c%22%3A%22lifestyle%2Centertainment%2Cnews%22%2C%22t%22%3A1788051072154%7D%26collegeFantasySportsPrefs%3D%7B%22clicked%22%3Afalse%2C%22views%22%3A1%7D; ySID=v=1&d=Q6oixakZog--; gpp=DBAA; gpp_sid=-1; _ga_B40QGCQW3G=GS2.1.s1788719628$o1$g1$t1788719643$j56$l0$h0; _ga_B3M8S64F9H=GS2.1.s1788719629$o1$g1$t1788719643$j46$l0$h0; A1S=d=AQABBMiIg2kCEBx_lDqWh1154c6DVTJIlREFEgEACAK-lGrNatwy0iMA7eMDAAcIyIiDaTJIlREID8YxE3vW5VsskSzECeT-xAkBBwoB7A&S=AQAAAp9sgJNDwMg-RXeAVbCFKcU; OTHD=g=ACE09DDA4051E75195B1CD22A43E1AFA1FC754C58EE7BA5EBA81F0BF1E8E8428&s=C617F1BE4550CA9A1A782168937DBC6A912E7BA5BCF5D780F1DBAD45C343042F&b=bid-135a869ko7268&j=us&bd=fc4533cca2d0b451d8d876697617435e&gk=x1a9z-qJCjBBcc&sk=x1a9z-qJCjBBcc&bk=x1a9z-qJCjBBcc&iv=1E168C36CAC1E9AADF20A595686FF755&v=1&u=0; SPT=d=zDUPxHiwjB1u2FEgvfu2CNUjw4RHmyXvo1fY6Y5ePYaWGr6BfolyCnabcD0fDtoTr30ONk8O3e2bvc1_95NRpo4gC6xACLitk1ze3ULHe8lUIRtAzYS.yXFFS9Y-&v=1; connectId=%7B%22puid%22%3A%22db123056943b885babda0bb02675d01b61273818236e15de6196e3662239a584%22%2C%22vmuid%22%3A%22gyXMt9dFHIvGFtGHZTte314nAgOGnMi3k9uqnip8y8Mt88xa3UfyU1QT26mhzrIUEHWDKgXhEYluGxH2eq0M_g%22%2C%22connectid%22%3A%22gyXMt9dFHIvGFtGHZTte314nAgOGnMi3k9uqnip8y8Mt88xa3UfyU1QT26mhzrIUEHWDKgXhEYluGxH2eq0M_g%22%2C%22connectId%22%3A%22gyXMt9dFHIvGFtGHZTte314nAgOGnMi3k9uqnip8y8Mt88xa3UfyU1QT26mhzrIUEHWDKgXhEYluGxH2eq0M_g%22%2C%22ttl%22%3A86400000%2C%22lastSynced%22%3A1788743703229%2C%22lastUsed%22%3A1788743703229%7D; GUCS=Aayk8VIk; _ga_MQBT164R4Q=GS2.1.s1788743832$o1$g1$t1788743845$j47$l0$h0; _ga=GA1.1.601588961.1788129234; tbla_id=759ff25d-2cb5-4f6c-bc56-e5770ac79cab-tuct107d0e4b; axids=gam=y-kxuT48JG2uKE0x8D_BQz67Fez0k0BNxCmdtAzFAL_py0EqOw9Q---A&dv360=eS1rM2pORU8xRTJ1SEpqaDR0ZXNUVFJvZS5wX1VENF9JVDh4Zlc4YVNCS0w2SjlQdWZWcEZjY3VPWm1BeUdEa0xpZGNDZX5B&ydsp=y-D.lFk.BE2uKVjP_UUT6fwIW5_PeWpkWLfXPH4dTKMJXmS.yLJJNm.NOGp7keorU4SsbC~A&tbla=y-1KUbVsJG2uJ2iEdCpbZczbgVQ6jq5Z5XNUL.B_IUJHDEu8a1SA---A; _ga_8CSZHGJ8KX=GS2.1.s1788743706$o3$g1$t1788744246$j59$l0$h0; __gads=ID=b44a6747865b84bd:T=1788744360:RT=1788744360:S=ALNI_MaNM9lVeeYIJv2iltm2ULSxI2BBcA; __gpi=UID=000013a0bfff3cce:T=1788744360:RT=1788744360:S=ALNI_Mbz9-wHHDEdf9jycM6hISircFacXw; __eoi=ID=481a8aa32d3345c3:T=1788744360:RT=1788744360:S=AA-AfjZt1H0Yj3SYNPKdp2OaZ6Ng"""

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Cookie': cookie_str,
    'Accept-Language': 'en-US,en;q=0.9'
}

print("Fetching URL...")
res = requests.get(url, headers=headers, allow_redirects=False)
print("Status:", res.status_code)
if res.status_code == 302:
    print("Redirect Location:", res.headers.get('Location'))
else:
    with open('draft_page.html', 'w') as f:
        f.write(res.text)
    print("Saved draft_page.html!")
