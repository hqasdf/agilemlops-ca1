import unittest
from unittest.mock import patch
from src.pbi10_command import execute_pipeline


class PipelineCommandTests(unittest.TestCase):
    def test_all_components_wired_and_connection_closed(self):
        class Connection:
            closed=False
            def close(self): self.closed=True
        connection=Connection(); calls=[]
        def prepare(urls): calls.append('download'); return {'cleaned':'batch'}
        def load(conn, cleaned, load_table=None): calls.append('load'); return {'orders':{'inserted':1}}
        def read(conn): calls.append('read'); return 'updated'
        def features(tables,sealed,new,current): calls.append('features'); return 'prepared'
        def train(member,prepared,batch,uri): calls.append(member); return [{'run_id':member,'model_uri':'model'}]
        with patch('src.pbi10_command.prepare_downloaded_batch',prepare), patch('src.pbi10_command.load_cleaned_batch',load), patch('src.pbi10_command.read_tables',read), patch('src.pbi10_command.prepare_retraining',features), patch('src.pbi10_command.train_member',train), patch('src.pbi10_command.batch_order_keys',return_value={'new'}), patch('src.pbi10_command.store_new_keys'):
            result=execute_pipeline(connection,{},'batch1','http://server',{'sealed'},set(),'ledger')
        self.assertEqual(result['status'],'success')
        self.assertEqual(calls,['download','load','read','features','han_qian','nithish','ephraim'])
        self.assertTrue(connection.closed)

    def test_rejects_sealed_key_before_database_write(self):
        class Connection:
            def close(self): pass
        with patch('src.pbi10_command.prepare_downloaded_batch',return_value={'cleaned':'batch'}), patch('src.pbi10_command.batch_order_keys',return_value={'sealed'}), patch('src.pbi10_command.load_cleaned_batch') as load:
            result=execute_pipeline(Connection(),{},'batch1','http://server',{'sealed'},set(),'ledger')
        self.assertEqual(result['status'],'failed')
        load.assert_not_called()

    def test_rejects_missing_manifest_before_database_write(self):
        class Connection:
            def close(self): pass
        with patch('src.pbi10_command.prepare_downloaded_batch',return_value={'cleaned':'batch'}), patch('src.pbi10_command.batch_order_keys',return_value={'new'}), patch('src.pbi10_command.load_cleaned_batch') as load:
            result=execute_pipeline(Connection(),{},'batch1','http://server',set(),set(),'ledger')
        self.assertEqual(result['status'],'failed')
        load.assert_not_called()

