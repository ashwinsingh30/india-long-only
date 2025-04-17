import json
import os

import requests
from requests.auth import HTTPBasicAuth

from config.ConfiguredLogger import get_logger

log = get_logger(os.path.basename(__file__))


class CiqServiceException(Exception):
    pass


class CapIQClient:
    _endpoint = 'https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/v4/clientservice.json'
    _headers = {'Content-Type': 'application/json', 'Accept-Encoding': 'gzip,deflate',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/80.0.3987.149 Safari/537.36'}
    _verify = True
    _username = None
    _password = None
    request_count = 0

    def __init__(self, username, password, verify=True, debug=False):
        assert username is not None
        assert password is not None
        assert verify is not None
        assert debug is not None
        self._username = username
        self._password = password
        self._verify = verify
        self._debug = debug

    def gdsp(self, identifiers, mnemonics, return_keys, properties=None):
        return self.make_request(identifiers, mnemonics, return_keys, properties, "GDSP", False)

    def gdspv(self, identifiers, mnemonics, return_keys, properties=None):
        return self.make_request(identifiers, mnemonics, return_keys, properties, "GDSPV", False)

    def gdst(self, identifiers, mnemonics, return_keys, start_date=None, end_date=None, frequency=None,
             properties=None):
        # properties or the start_date and frequency must be set
        if not properties:
            properties = []
            for i in range(0, len(mnemonics)):
                properties.append({})
        for p in properties:
            if frequency:
                p["FREQUENCY"] = frequency
            if start_date:
                p["STARTDATE"] = start_date
            if end_date:
                p["ENDDATE"] = end_date
        return self.make_request(identifiers, mnemonics, return_keys, properties, "GDST", True)

    def gdshe(self, identifiers, mnemonics, return_keys, start_date=None, end_date=None, properties=None):
        if not properties:
            properties = []
            for i in range(0, len(mnemonics)):
                properties.append({})
        for p in properties:
            if start_date:
                p["STARTDATE"] = start_date
            if end_date:
                p["ENDDATE"] = end_date
        return self.make_request(identifiers, mnemonics, return_keys, properties, "GDSHE", True)

    def gdshv(self, identifiers, mnemonics, return_keys, start_date=None, end_date=None, properties=None):
        if not properties:
            properties = []
            for i in range(0, len(mnemonics)):
                properties.append({})
        for p in properties:
            if start_date:
                p["STARTDATE"] = start_date
            if end_date:
                p["ENDDATE"] = end_date
        return self.make_request(identifiers, mnemonics, return_keys, properties, "GDSHV", False)

    def gdsg(self, identifiers, group_mnemonics, return_keys, properties=None):
        return self.make_request(identifiers, group_mnemonics, return_keys, properties, "GDSG", False)

    def get_request_count(self):
        return self.request_count

    def make_request(self, identifiers, mnemonics, return_keys, properties, api_function_identifier,
                     multiple_results_expected):
        req_array = []
        returnee = {}
        tmp_request_count = 0
        for identifier in identifiers:
            for i, mnemonic in enumerate(mnemonics):
                req_array.append({"function": api_function_identifier, "identifier": identifier, "mnemonic": mnemonic,
                                  "properties": properties[i] if properties else {}})
                tmp_request_count += 1
        req = {"inputRequests": req_array}
        response = requests.post(self._endpoint, headers=self._headers, data=json.dumps(req),
                                 auth=HTTPBasicAuth(self._username, self._password), verify=self._verify)
        print(response.json())
        for return_index, ret in enumerate(response.json()['GDSSDKResponse']):
            identifier = ret['Identifier']
            if identifier not in returnee:
                returnee[identifier] = {}
            returned_properties = {}
            if "Properties" in ret:
                returned_properties = ret['Properties']
            if ret['ErrMsg']:
                raise RuntimeError('Cap IQ error for ' + identifier + ' + ' + ret['Mnemonic'] + ' query: ' +
                                   ret['ErrMsg'])
            else:
                for i_m, h_m in enumerate(ret["Headers"]):
                    if multiple_results_expected:
                        returnee[identifier][
                            self.get_return_key(ret['Mnemonic'])] = []
                        for row in ret["Rows"]:
                            returnee[identifier][
                                self.get_return_key(ret['Mnemonic'])
                            ].append(row['Row'])
                    else:
                        returnee[identifier][
                            self.get_return_key(ret['Mnemonic'])
                        ] = ret['Rows'][i_m]['Row'][0]
        return returnee

    @staticmethod
    def get_return_key(mnemonic):
        return mnemonic.lower()

    def request_point_in_time_ltm_data(self, ciq_client, identifiers, mnemonics, date):
        properties = {"periodType": "IQ_LTM",
                      "consolidatedFlag": "CON",
                      "currencyId": "INR",
                      "restatementTypeId": "LRP",
                      "filingMode": "F",
                      "AsOfDate": str(date.strftime("%m/%d/%Y")),
                      "metaDataTag": "AsOfDate"}
        property_list = [properties] * len(mnemonics)
        return ciq_client.gdsp(identifiers, mnemonics, mnemonics, property_list)

    def request_point_in_time_data_for_period_type(self, ciq_client, identifiers, mnemonics, date, period_type):
        properties = {"periodType": period_type,
                      "consolidatedFlag": "CON",
                      "currencyId": "INR",
                      "restatementTypeId": "LRP",
                      "filingMode": "F",
                      "AsOfDate": str(date.strftime("%m/%d/%Y")),
                      "metaDataTag": "AsOfDate"}
        property_list = [properties] * len(mnemonics)
        return ciq_client.gdsp(identifiers, mnemonics, mnemonics, property_list)

    def request_point_in_time_data_no_props(self, ciq_client, identifiers, mnemonics, date):
        properties = {"AsOfDate": str(date.strftime("%m/%d/%Y")),
                      "metaDataTag": "AsOfDate"}
        property_list = [properties] * len(mnemonics)
        return ciq_client.gdsp(identifiers, mnemonics, mnemonics, property_list)

    def request_point_in_time_data_for_period_type_date_range(self, ciq_client, identifiers, mnemonics,
                                                              start_date, end_date, period_type):
        properties = {"periodType": period_type,
                      "consolidatedFlag": "CON",
                      "currencyId": "INR",
                      "restatementTypeId": "LRP",
                      "filingMode": "F"}
        property_list = [properties] * len(mnemonics)
        return ciq_client.gdshe(identifiers, mnemonics, mnemonics,
                                start_date=str(start_date.strftime("%m/%d/%Y")),
                                end_date=str(end_date.strftime("%m/%d/%Y")),
                                properties=property_list,
                                )

    def request_point_in_time_data_no_props_date_range(self, ciq_client, identifiers, mnemonics, date):
        properties = {"AsOfDate": str(date.strftime("%m/%d/%Y")),
                      "metaDataTag": "AsOfDate"}
        property_list = [properties] * len(mnemonics)
        return ciq_client.gdsp(identifiers, mnemonics, mnemonics, property_list)
