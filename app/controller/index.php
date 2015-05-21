<?php

namespace app\controller;
use app\controller\base\controller, app\request;
use library\layout\components\template;
use library\layout\elements\block;
use library\layout\elements\button;
use library\layout\elements\form;
use library\layout\elements\group;
use library\layout\elements\input;
use library\event\event;
use library\layout\elements\script;

class index extends controller{

    public function __construct($action,$query) {
        parent::__construct($action,$query);
    }

    public function index(){
        //Creates a new div element
        $container = new group(array("container","theme-showcase"));
        //Creates a new button element
        $button3 = new button("Don't Click me","button");
        //Sets the id of the element
        $button3->setId("button","carro");
        //Loads a static file into a template object
        $template = new template("index/index");
        //Adds the button element to the template
        $template->addElement($button3);
        //Creates a new form
        $form = new form("/index/index");
        //Creates a new group element
        $group = new group();
        //Creates a new input element
        $input = new input("text","Hello world","Escreva:");
        //Creates a new group element
        $group2 = new group();
        //Creates a new button
        $button = new button("Click me","button");
        //Adds the input element to the group element
        $group->addChild($input);
        //Adds the button element to the group element
        $group2->addChild($button);
        //Adds the template to the form
        $form->addChild($template);
        //Adds the group element to the form
        $form->addChild($group);
        //Adds the group element to the form
        $form->addChild($group2);
        //Adds the form element to the container
        $container->addChild($form);
        //Adds the container to the page layout
        $this->layout->addChild($container);
        //Binds the button element to a event and a callback function
        $button->bind('click', function($button){
            $button->addClassName("red");
            $button->changeValue("Clicked");
            $group2 = $button->getParent();
            $button2 = new button('Botao2','button');
            $group2->addChild($button2,$group2->getUid(),"prepend");
        });
        //Binds the button element to a event and a callback function
        $button3->bind('click', function($button){
            $button->addClassName("red");
            $button->changeValue("I told you not to click me");
            $group = $button->getParent();
            $button2 = new button('Botao Top','button');
            $group->addChild($button2,$group->getUid(),"append");
        });
        
        if(request::isAjax()){
            //triggers the event
            event::trigger($this->query["uid"], $this->query["event"]);
            //returns a json object
            echo script::getResponse();
            exit;
        } else {
            //returns the requested page
            $this->html();
        }
    }
    
    
    

}
