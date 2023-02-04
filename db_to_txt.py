import mysql.connector
import os
import asyncio

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="corpus"
)


async def main():
    mycursor = mydb.cursor()

    mycursor.execute("SELECT category,count(*) FROM `allposts` group by category order by count(*)")
    cresult = mycursor.fetchall()
    k = 0
    for c in cresult:
        k = k + 1
        mycursor.execute("SELECT CONVERT(title USING utf8), CONVERT(text USING utf8) FROM allposts where category='" + c[0] + "'")

        myresult = mycursor.fetchall()
        i = 0
        for x in myresult:
            i = i + 1
            print(k, i, c[1])
            f = open("C:/Users/E-MaxPCShop/Desktop/corpus_utf8/" + c[0] + "/" + str(i) + ".txt", "w", encoding="utf8")
            f.write(x[0].strip() + "\n\n" + x[1])
            f.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
