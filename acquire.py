import os
import pandas as pd

def zillow_2017_data():
    '''
    This function uses a SQL query to access the Codeup MySQL database and join 
    together all the relevant data from the zillow database.
    The data obtained includes all properties in the dataset which had a transaction in 2017.
    The function caches a csv in the local directory for later use. 
    '''
    # establish a filename for the local csv
    filename = 'zillow.csv'
    # check to see if a local copy already exists. 
    if os.path.exists(filename):
        print('Reading from local CSV...')
        # if so, return the local csv
        return pd.read_csv(filename)
    # otherwise, pull the data from the database:
    # establish database url
    url = env.get_db_url('zillow')
    # establish query
    sql = '''
            SELECT prop.*,
                   pred.logerror,
                   const.typeconstructiondesc,
                   arch.architecturalstyledesc,
                   land.propertylandusedesc,
                   heat.heatingorsystemdesc,
                   air.airconditioningdesc, 
                   bldg.buildingclassdesc,
                   story.storydesc
              FROM properties_2017 prop
                JOIN predictions_2017            pred  USING(parcelid)
                LEFT JOIN typeconstructiontype   const USING(typeconstructiontypeid)
                LEFT JOIN architecturalstyletype arch  USING(architecturalstyletypeid)
                LEFT JOIN propertylandusetype    land  USING(propertylandusetypeid)
                LEFT JOIN heatingorsystemtype    heat  USING(heatingorsystemtypeid)
                LEFT JOIN airconditioningtype    air   USING(airconditioningtypeid)
                LEFT JOIN buildingclasstype      bldg  USING(buildingclasstypeid)
                LEFT JOIN storytype              story USING(storytypeid)
              WHERE pred.transactiondate LIKE "2017%%"
                AND pred.transactiondate in (
                                             SELECT MAX(transactiondate)
                                               FROM predictions_2017
                                               GROUP BY parcelid
                                             )
                AND prop.latitude IS NOT NULL
                AND prop.longitude IS NOT NULL;
            '''
    print('No local file exists\nReading from SQL database...')
    # query the database and return the resulting table as a pandas dataframe
    df = pd.read_sql(sql, url)
    # save the dataframe to the local directory as a csv
    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    # return the resulting dataframe
    return df

