import ESL

con = ESL.ESLconnection("127.0.0.1", "8021", "ClueCon")

if con.connected():
    print("Connected to FreeSWITCH")
else:
    print("Connection failed")
