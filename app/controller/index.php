
 <?php

namespace app\controller;
use app\controller\base\controller, app\request;
use library\dom\elements\components\button;
use library\dom\elements\components\block;
use library\dom\elements\components\link;
use library\dom\elements\components\form;
use library\dom\elements\components\select;
use library\dom\elements\components\option;
use library\dom\structures\dropdown;
use library\dom\structures\buttonGroup;
use library\dom\structures\inputGroup;
use library\dom\structures\sTable;
use library\dom\structures\alert;
use library\dom\elements\components\script;
use library\dom\elements\components\text;
use library\dom\elements\components\title;
use library\dom\structures\template;
use library\dom\elements\layout;
use library\dom\dom;
use library\dom\javascript;
use library\event\event;
use app\model\file;


class index extends controller{
    private $src = array("https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js","/js/bootstrap.min.js","/js/script.js");


    public function __construct($action,$query) {
        parent::__construct($action,$query);
    }
    
    public function teste(){
        if(request::isAjax()){
            $event = new event($_GET['event'],$_GET['uid']);
            try{
                $event->trigger();
            } catch (\Exception $ex) {
                $alert = new alert();
                $alert->create($ex->getMessage());
                dom::getElementByTagName('body')->addComponent($alert->save());
            }
            echo javascript::getResponse();
        } else {
            dom::init();
            //load head from template file
            $template = new template("head","head");
            //add head title
            $title = new title("Pagina de teste");
            $template->addComponent($title,"title");
            //create head component
            $head = $template->save();

            //add head component to the document
            dom::add($head);
            
            //create new layout component
            $layout = new layout();
            $layout->setLayoutType("fixed");
            
            //create new div
            $row = new block("div");
            $row->addClass("row");
            
            //add div to the layout
            $layout->addComponent($row);
            
            //create new div
            $col8 = new block("div");
            $col8->addClass("col-md-8");
            
            //add div to the previous div
            $row->addComponent($col8);
            
            $form = new form("/index/teste");
            
            //create new inputGroup structure
            $inputGroup = new inputGroup();
            $inputGroup->create("price","%","right");
            
            //add the inputGroup structure to the form
            $form->addComponent($inputGroup->save());
            
            
            $select = new select("csv");
            $select->addOption(new option("Select file..."));
            
            //read files with extension "csv" from the given directory
            $csv = file::getFiles(FILES_PATH.DS,"csv");
            
            foreach($csv as $file){
                $option = new option($file);
                $option->setValue($file);
                $select->addOption($option);
            }
            
            $form->addComponent($select);
            
            $submit = new button("Load Table");
            //add Event Listener to the component
            $submit->addEventListener('submit', array($this,"submit"), array($submit));
            
            
            $form->addComponent($submit);
            
            $col4 = new block("div");
            $col4->addClass("col-md-4");
            $row->addComponent($col4);
            
            $p = new block("p");
            $texto = new text("Hello World");
            $p->addComponent($texto);
            
            $col8->addComponent($p);
            $col8->addComponent($form);
            
            $buttonGroup = new buttonGroup('btn-group-justified');
            
            $button = $buttonGroup->addButton("Create table");
            $button->addEventListener('click', array($this,"ajax"),array($button));
            
            $dropdown = new dropdown("btn-group");
            $dropdown->create("Lista");
            $list = array("Carro","Casa","Escola");
            foreach($list as $item){
                $dropdown->addHeader("Item");
                $dropdown->addLink(new link($item));
            }

            $buttonGroup->nest($dropdown);
            
            $col4->addComponent($buttonGroup->save());

            dom::add($layout);
            
            //Add js files
            $this->exScript();
            
            //render the page
            dom::save();
        }
    }
    private function exScript(){
        foreach($this->src as $src){
            dom::add(new script($src));
        }
    }

    public function ajax($button){
        assert(request::isAjax());
        //create sTable structure
        $stable = new sTable();
        //Add rows and columns from array
        $stable->create(array(array("Carro","12","123"),array("Carro","12","123"),array("Carro","12","123")));
        $stable->addHeader(array("Carro","Casa","Total"));
        //create table component to be appended to the page
        $table = $stable->save();
        $table->addClass("table-striped");
        /*
         * $button refers to button that was clicked and initiated the event. "closest" and "find"
         * functions work like they do in jQuery
         */
        $button->closest('div.container')->find('div.col-md-8')->addComponent($table);
        // Remove the event listener from the component so it will not be triggered again
        $button->removeEventListener('click');
        
    }

    
    public function submit($button){
        assert(request::isAjax());
        $csv = file::getFiles(FILES_PATH.DS,"csv");
        $data = $_GET['csv'];
        if(!in_array($data,$csv)){
            throw new \Exception("File not supported");
        } 
        $button->closest("div")->find("button")->addClass("red");
        $stable = new sTable();
        // Read CSV file and create a sTable structure with it
        $stable->readCSV(FILES_PATH.DS.$data,true);
        // Create a table component to be appended to the page
        $table = $stable->save();
        $table->addClass("table-striped");
        $button->closest("div")->find("button")->closest("div")->addComponent($table);
        $button->attr("type","button");
        $button->removeEventListener('submit');
        
    }
    
    
    
    

}
