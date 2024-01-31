package at.clifford.checklist;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.LineNumberReader;
import java.util.ArrayList;

import android.app.ListActivity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CheckedTextView;
import android.widget.ListView;
import android.widget.Toast;
import android.widget.AdapterView.OnItemClickListener;

public class CheckList extends ListActivity
{
	private static class Line {
		public String name;
		public boolean status;
		public Line(String n, boolean s) {
			name = n;
			status = s;
		}
		public String toString() {
			return name;
		}
		public boolean isChecked() {
			return status;
		}
	};
	private class LineArrayAdaptor extends ArrayAdapter<Line> {
		Line[] objects;
		public LineArrayAdaptor(Context context, Line[] objects) {
			super(context, android.R.layout.simple_list_item_checked, objects);
			this.objects = objects;
		}
		public View getView(int pos, View inView, ViewGroup parent) {
			CheckedTextView v = (CheckedTextView) super.getView(pos, inView, parent);
			// v.setEnabled(objects[pos].isChecked());
			v.setChecked(objects[pos].isChecked());
			return v;
		}
	};

	String fileName;
	
	private Handler saveHandler = new Handler();
	private Runnable saveTask = new Runnable() {
		public void run() {
			if (isDirty)
				saveFile();
			
		}
	};
	
	Line[] mydata = new Line[0];
	public boolean isDirty = false;

	int PICK_OPEN_REQUEST_CODE = 0;
	int PICK_SAVE_AS_REQUEST_CODE = 1;
	
	MenuItem menu_open_file;
	MenuItem menu_save_as;
	
	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		menu_open_file = menu.add("Open File...");
		menu_save_as = menu.add("Save as...");
	    return true;
	}
	
	@Override
	public boolean onOptionsItemSelected(MenuItem item)
	{
		if (item == menu_open_file)
		{
			Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
			intent.putExtra("title", "Open CSV file");
            intent.setType("file/*");
            try {
            	startActivityForResult(intent, PICK_OPEN_REQUEST_CODE);
            } catch (android.content.ActivityNotFoundException ex) {
    			Toast.makeText(getApplicationContext(),
    					"No Activity for ACTION_GET_CONTENT file/*",
    					Toast.LENGTH_SHORT).show();
            }
			return true;
		}
		if (item == menu_save_as)
		{
			Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
			intent.putExtra("title", "Save as CSV file");
            intent.setType("file/*");
            try {
            	startActivityForResult(intent, PICK_SAVE_AS_REQUEST_CODE);
            } catch (android.content.ActivityNotFoundException ex) {
    			Toast.makeText(getApplicationContext(),
    					"No Activity for ACTION_GET_CONTENT file/*",
    					Toast.LENGTH_SHORT).show();
            }
			return true;
		}
		return super.onOptionsItemSelected(item);
	}

	private boolean readFile()
	{
		if (fileName == null)
			return false;
	
		try
		{
			ArrayList<Line> buf = new ArrayList<Line>();
			
			FileReader reader = new FileReader(fileName);
			LineNumberReader lr = new LineNumberReader(reader);
			for (String line = lr.readLine(); line != null; line = lr.readLine()) {
				int pos = line.lastIndexOf(",");
				if (pos >= 0) {
					String name = line.substring(0, pos);
					String val = line.substring(pos+1);
					buf.add(new Line(name, val.indexOf("1") >= 0));
				}
			}
			lr.close();
			reader.close();

			mydata = buf.toArray(mydata);
			setListAdapter(new LineArrayAdaptor(this, mydata));
			// ((LineArrayAdaptor)getListAdapter()).notifyDataSetChanged();
		}
		catch (Exception ex)
		{
			Toast.makeText(getApplicationContext(),
					"Failed to open " + fileName,
					Toast.LENGTH_SHORT).show();
			fileName = null;
			return false;
		}

		isDirty = false;
		Toast.makeText(getApplicationContext(),
				"Opened: " + fileName, Toast.LENGTH_SHORT).show();
		
		return true;
	}
	
	private boolean saveFile()
	{
		if (fileName == null)
			return false;

		try
		{
			FileWriter writer = new FileWriter(fileName);
			for (int i=0; i<mydata.length; i++) {
				String line = mydata[i].name + "," +
					(mydata[i].status ? "1" : "#") + "\r\n";
				writer.write(line);
			}
			writer.close();
		}
		catch (Exception ex)
		{
			Toast.makeText(getApplicationContext(),
					"Failed to open " + fileName,
					Toast.LENGTH_SHORT).show();
			fileName = null;
			return false;
		}

		isDirty = false;
		Toast.makeText(getApplicationContext(),
				"Saved: " + fileName, Toast.LENGTH_SHORT).show();
		
		return true;
	}
	
	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent intent)
	{
		if (requestCode == PICK_OPEN_REQUEST_CODE && resultCode == RESULT_OK) {
			fileName = intent.getData().getPath();
			readFile();
		}
		if (requestCode == PICK_SAVE_AS_REQUEST_CODE && resultCode == RESULT_OK) {
			fileName = intent.getData().getPath();
			saveFile();
		}
	}

	@Override
	public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        
        setListAdapter(new LineArrayAdaptor(this, mydata));
        
        ListView lv = getListView();
        lv.setTextFilterEnabled(true);
        
        lv.setOnItemClickListener(new OnItemClickListener() {
        	public void onItemClick(AdapterView<?> parent, View view,
                int position, long id)
        	{
            	mydata[position].status = !mydata[position].status;
            	((LineArrayAdaptor)parent.getAdapter()).notifyDataSetChanged();

            	isDirty = true;
            	saveHandler.removeCallbacks(saveTask);
            	saveHandler.postDelayed(saveTask, 2000);
            }
          });
        
        fileName = "/sdcard/checklist.csv";
        readFile();
    }
}
