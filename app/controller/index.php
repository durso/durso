<?php

namespace app\controller;

// just a simple mvc built upon CS50 PHP files 
use app\controller\base\controller, app\request;

// Load components (they are html tags)
use library\dom\elements\components\block;
use library\dom\elements\components\link;


// Load structures (they are wrappers for components)
use library\dom\structures\media;
use library\dom\structures\sTable;
use library\dom\structures\alert;
use library\dom\structures\template;


use library\dom\dom;
use library\dom\javascript;
use library\event\event;
use library\api\google\places;



class index extends controller{
   
    public function __construct($action,$query) {
        parent::__construct($action,$query);
    }
    
    public function index(){
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
            $template->create(TEMPLATE_PATH.DS."index.html");
            dom::add($template->save());
            
            // Gets the component that has id equal to navbar
            $navbar = dom::getElementById("navbar");
            // Create a new list component
            $li = dom::createElement("li");
            // Create a new anchor component with the text value of Click here
            $a = new link("Click here");
            /* 
             * Add an Event Listener called location. This event will get the user location
             * once the anchor component is clicked 
             * Other events are: click and submit
             * addEventListener has 3 parameters: the first one is the the name of the
             * event explained above. The second one is the function to be called once
             * the event is triggered. In this case it will call the method location from 
             * this object. The last parameter is optional, it is an array with arguments
             * you might want to access from the function passed as the second parameter
             */
            $a->addEventListener("location", array($this,"location"), array($a));
            // Add the anchor component to the list
            $li->addComponent($a);
            /*
             *  The method find uses a css selector to locate and return a component
             *  Once the component is returned, I added the list component to it
             */   
            $navbar->find("ul.nav")->addComponent($li);
            // output the DOM tree as a string
            dom::save();
        }
        
        
    }

    // just test
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
        $button->closest('nav.navbar')->siblings('div')->find('div.starter-template')->append($table);
        // Remove the event listener from the component so it will not be triggered again
        dom::getElementById("starter")->find(".lead")->clear();
        $button->removeEventListener('click');
        
    }

    // just test
    public function submit($button){
        assert(request::isAjax());
    

        $stable = new sTable();
        // Read CSV file and create a sTable structure with it
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

