<?php

namespace app\controller;

// just a simple mvc framework
use app\controller\base\controller, app\request;

// Load components (they are html tags)
use library\dom\elements\components\block;
use library\dom\elements\components\link;


// Load structures (they are wrappers for components)
use library\dom\structures\media;
use library\dom\structures\sTable;
use library\dom\structures\alert;
use library\dom\structures\template;

use library\dom\chart\gChart;

use library\dom\dom;
use library\dom\javascript;
use library\event\event;
use library\api\google\places;



class index extends controller{
   
    public function __construct($action,$query) {
        parent::__construct($action,$query);
    }
    
    public function events(){
        // check if request is an Ajax request
        if(request::isAjax()){
            // set up a new event
            $event = new event($_GET['event'],$_GET['uid']);
            try{
                // triggers the event
                $event->trigger();
            } catch (\Exception $ex) {
                $alert = new alert();
                $alert->create($ex->getMessage());
                dom::getElementByTagName('body')->addComponent($alert->save());
            }
            // output json response
            echo javascript::getResponse();
        } else {
            // initializes DOM tree
            dom::init();
            
            /*
             * create a new structure (wrapper of components) from a file and then add 
             * the result to the DOM tree
             */
            $template = new template();
            $template->create(TEMPLATE_PATH.DS."index2.html");
            dom::add($template->save());
            
            // Gets the component that has id equal to navbar
            $navbar = dom::getElementById("navbar");
            // Create a new list component
            $li = dom::createElement("li");
            // Create a new anchor component with the text value of Don't Forget to Click here
            $a = new link("Don't forget to Click here");
            /* 
             * Add an Event Listener called location. This event will get the user location
             * once the anchor component is clicked 
             * Other events are: click and submit
             * addEventListener has 3 parameters: the first one is the the name of the
             * event explained above. The second one is the function to be called once
             * the event is triggered. In this case it will call the method location from 
             * this object. The last parameter is optional, it is an array with arguments
             * you might want to access from the function passed as the second parameter
             * PLEASE REMEMBER TO CLICK ON THE LINK "CLICK HERE" IF YOU WANT TO SEE HOW THE EVENT WORKS
             */
            $a->addEventListener("location", array($this,"location"), array($a));
            // Add the anchor component to the list
            $li->addComponent($a);
            /*
             *  The method find uses a css selector to locate and return a component
             *  Once the component is returned, the list component is added to it
             */   
            $navbar->find("ul.nav")->addComponent($li);
            // output the DOM tree as a string
            dom::save();
        }
    }
    
    public function index(){
        // initializes DOM tree
        dom::init();

        /*
         * create a new structure (wrapper of components) from a file (in this case, bootstrap SB 2 dashboard) and then add 
         * the result to the DOM tree
         */
        $template = new template();
        $template->create(TEMPLATE_PATH.DS."index.html");
        dom::add($template->save());

        // select the head component in the dom tree
        $head = dom::getElementByTagName("head");

        // uses the method find to select the title component and then changes its value using the method innerHTML
        $head->find("title")->innerHTML(dom::createTextNode("Testing page"));

        /*
         * Create a new Google Chart component by passing a name, the type of chart and the id of the html element that will contain the chart
         * Could have created a private method to create and return all the charts below so the repetition would be avoided, but my goal is to show how the gChart class works
         */
        $goals = new gChart("goals","LineChart","gChart1");
        $goals->readCSV("gols.csv");
        $goals->addOption("title", "Goals per season");
        $goals->removeGrids();
        $goals->textColor('white');
        $goals->prepare();

        $cars = new gChart("cars","LineChart","gChart2");
        $cars->readCSV("carros.csv");
        $cars->addOption("title", "Average car price");
        $cars->removeGrids();
        $cars->textColor('white');
        $cars->prepare();

        $houses = new gChart("houses","LineChart","gChart3");
        $houses->readCSV("casas.csv");
        $houses->addOption("title", "Average house price");
        $houses->removeGrids();
        $houses->textColor('white');
        $houses->prepare();


        $rent = new gChart("rent","LineChart","gChart4");
        $rent->readCSV("rent.csv");
        $rent->addOption("title", "Average rent price");
        $rent->removeGrids();
        $rent->textColor('white');
        $rent->prepare();

        $all = new gChart("all","LineChart","gChart5");
        $all->readCSV("all.csv");
        $all->addOption("title", "All areas");
        $all->prepare();

        //Add all the charts created above to the head component 
        $head->addComponent(gChart::save());

        //save and output the dom tree
        dom::save();  
    }

    // This code is not executed by this file as this method was not passed to the event listener
    public function ajax($wrapper){
        assert(request::isAjax());
        //create sTable structure
        $stable = new sTable();
        //Add rows and columns from array
        $stable->create(array(array("Carro","12","123"),array("Carro","12","123"),array("Carro","12","123")));
        $stable->addHeader(array("Carro","Casa","Total"));
        //create table component to be appended to the page
        $table = $stable->save();
        $table->addClass("table-striped");

        $wrapper->append($table);
        
    }

    // This code is not executed by this file as this method was not passed to the event listener
    public function submit($button){
        assert(request::isAjax());
    

        $stable = new sTable();
        // Read CSV remote file and create a sTable structure with it
        $stable->readCSV("https://extranet.who.int/tme/generateCSV.asp?ds=tbhivnonroutinesurv",true);
        // Create a table component to be appended to the page
        $table = $stable->save();
        $table->addClass("table-striped");
        $button->closest("body")->find("div.starter-template")->addComponent($table);
        $button->removeEventListener('click');
        
    }
     
    
    public function location($button){
        assert(request::isAjax());
        
        // Load an object to deal with google places api requests
        $places = new places();
        
        // Get the user location (latitude,longitude) to search for the keyword pub nearby
        // within a radius of 5000m
        $places->prepare($_GET['lat'],$_GET['long'],"pub",5000);
        
        // Executes the api call
        $results = $places->exec();
        // Get a component by id
        $main = dom::getElementById("starter");
        // Creates a new div component
        $div = new block("div");
        // Adds bootstrap grid classes to the component
        $div->addClass("col-md-6 col-md-offset-3");
        /*
         *  Add the div component to component with id equals to starter
         *  Using append instead of addComponent will add a fade in effect
         */
        $main->append($div);
        // Loop through the api result
        foreach($results as $item){
            /*
             * Creates a new media structure to wrap each result and then adds it to the div
             * component
             */
            $media = new media();
            $media->create($item["icon"], $item["name"], $item["vicinity"]);
            $div->addComponent($media->save());
        }

    }


}